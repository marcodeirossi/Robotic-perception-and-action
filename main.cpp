#include <QCoreApplication>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <librealsense2/rs.hpp>
#include <librealsense2-gl/rs_processing_gl.hpp>
#include <exception>
#include <iostream>
#include <stdio.h>
//#include </home/user/Software/librealsense/examples/measure/rs-measure.cpp>
//#include </home/user/librealsenseNew/librealsense/examples/sensor-control/api_how_to.h>
#include </home/user/librealsenseNew/librealsense/include/librealsense2/rsutil.h>
#include </home/user/librealsenseNew/librealsense/include/librealsense2/rs.hpp>
#include <cv-helpers.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

//define the size of the detected window of 300 X 300

const size_t inWidth      = 300;                                            //define the size of the detected window of 300 X 300
const size_t inHeight     = 300;
const float WHRatio       = inWidth / (float)inHeight;                      //define the ratio width/height
const float inScaleFactor = 0.007843f;                                      //define scale factor
const float meanVal       = 127.5;                                          //define men value
const char* classNames[]  = {"background",
                             "aeroplane", "bicycle", "bird", "boat",
                             "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse",
                             "motorbike", "person", "pottedplant",
                             "sheep", "sofa", "train", "tvmonitor"};        //define the possible calles that the dnn can detect

int main(int argc, char** argv) try                                         //start of the itetation end of the declaration part and start of the real process
{
    using namespace std;
    using namespace cv;                                                     //library to be used untill line 57
    using namespace cv::dnn;
    using namespace rs2;

    Net net = readNetFromCaffe("/home/user/librealsenseNew/librealsense/wrappers/opencv/dnn/build/MobileNetSSD_deploy.prototxt",
                               "/home/user/librealsenseNew/librealsense/wrappers/opencv/dnn/build/MobileNetSSD_deploy.caffemodel");

    //start streaming from Intel RealSense Camera
    pipeline pipe;                                                          //define pipeline to start the d435
    auto config = pipe.start();                                             //auto configuration to get started
    auto profile = config.get_stream(RS2_STREAM_COLOR)
                         .as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

    Size cropSize;
    if (profile.width() / (float)profile.height() > WHRatio)                //start if actual ratio is bigger than the WHratio
    {
        cropSize = Size(static_cast<int>(profile.height() * WHRatio),profile.height());   //rescale the height
    }
    else
    {
        cropSize = Size(profile.width(),static_cast<int>(profile.width() / WHRatio));     //rescale the width
    }

    Rect crop(Point((profile.width() - cropSize.width) / 2,
                    (profile.height() - cropSize.height) / 2),
              cropSize);

    const auto window_name = "Display Image";                               //define windows_name
    namedWindow(window_name, WINDOW_AUTOSIZE);

    while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();
        // Make sure the frames are spatially aligned
        data = align_to.process(data);

        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();

        // If we only received new depth frame, but the color did not update, continue
        static int last_frame_number = 0;
        if (color_frame.get_frame_number() == last_frame_number) continue;
        last_frame_number = color_frame.get_frame_number();

        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
        auto depth_mat = depth_frame_to_meters(pipe, depth_frame);                          //define depth_math as function of depth_frame_to_meters

        Mat inputBlob = blobFromImage(color_mat, inScaleFactor,
                                      Size(inWidth, inHeight), meanVal, false);             //Convert Mat to batch of images
        net.setInput(inputBlob, "data");                                                    //set the network input
        Mat detection = net.forward("detection_out");                                       //compute output

        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        // Crop both color and depth frames
        color_mat = color_mat(crop);
        depth_mat = depth_mat(crop);

        float confidenceThreshold = 0.8f;                                                     //define the condifence threshold / minimum accuracy required at 80%
        for(int i = 0; i < detectionMat.rows; i++)                                            //for cycle
        {
            float save[i][3];
            uchar histogram[i][3];

            float confidence = detectionMat.at<float>(i, 2);                                  // define the actual real confidence

            if(confidence > confidenceThreshold)                                              //if cycle if we are sure that the detection is ?accurate? at least at 80% as defined before
            {
                size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

                if(classNames[objectClass] == "person")
                {
                                                                                              //define the two opposit points in right top and bottom left to define the rectangle
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * color_mat.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * color_mat.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * color_mat.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * color_mat.rows);

                                       //define the rectangle as object from one point LB (xlb,ylb) with the lenght of 2 lines on x and on y (xrt-xlt) or (xrt-xlb)
                Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),     //try also with (xrt-xlt) and also similarly in the following line
                            (int)(yRightTop - yLeftBottom));

                object = object  & Rect(0, 0, depth_mat.cols, depth_mat.rows);

                // Calculate mean depth inside the detection region
                // This is a very naive way to estimate objects depth
                // but it is intended to demonstrate how one might
                // use depht data in general
                Scalar m = mean(depth_mat(object));                                                         //define the depth as mean of the depth of each pixel contained in the rectangle

                        int xcenter = (xRightTop - (xLeftBottom / 2));
                        int ycenter = (yRightTop + (yLeftBottom / 2));
                                                                                  //we can use depth_mat in the center point  (SEARCH THE LIBRARY / DOCUMENTATION ON depth_frame_to_meters)
                std::ostringstream ss;
                ss << classNames[objectClass] << " ";                                                       //insert class name
                ss << std::setprecision(2) << m[0] << " meters away";                                       //output the mean distance fixed at mean distance meter away
                ss << ", centred in pixels ";                                                //inseted new line to say center point
                ss << xcenter ;
                ss << "  ";
                ss <<  ycenter;

                String conf(ss.str());

                rectangle(color_mat, object, Scalar(0, 255, 0));                                            //define rectangle color as green (0,255,0)
                int baseLine = 0;
                Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);            //the size of the white box is slightly bigger than the dimention of the text

                auto center = (object.br() + object.tl())*0.5;
                center.x = center.x - labelSize.width / 2;


                                                                                                            //define white rectangle for written part
                rectangle(color_mat, Rect(Point(center.x, center.y - labelSize.height),                     //it is centered in the center of the bounding box !!!!!!!!!!!!!!!!!!!!!!!
                    Size(labelSize.width, labelSize.height + baseLine)),                                    //size of the label written
                    Scalar(255, 255, 255), FILLED);                                                         //white color
                putText(color_mat, ss.str(), center,                                                        //the text should be centerd inside the box
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));                                          //the text is black

               float centrop[2];                                                                            //define the centerp in 2D as vector of 2 element
               centrop[0]=center.x;                                                                         //assing the first value the center x
               centrop[1]=center.y;                                                                         //assing the first value the center y
               float centrov[3];                                                                            //define the centerv in 3D as vector of 3 element
               float centrov1[3];                                                                           //define the centerv1 in 3D as vector of 3 element
               float centerdepth = depth_frame.get_distance(centrop[0], centrop[1]);                        //find out the distance of the center as get_distance of centerp


//   ad ora abbiamo le cordinate di profondit� data come media
//   il centro xcenter e ycenter come misure algebriche dagli angoli
//   centrop centrov con centerdepth del centro dato da loro


               //TO FIND OUT THE INTRISTIC PARAMETER

                  auto depth_stream = config.get_stream(RS2_STREAM_DEPTH)
                                               .as<rs2::video_stream_profile>();
                  auto resolution = std::make_pair(depth_stream.width(), depth_stream.height());
                  auto in = depth_stream.get_intrinsics();
                  auto principal_point = std::make_pair(in.ppx, in.ppy);
                  auto focal_length = std::make_pair(in.fx, in.fy);
                                  //auto matrix_parameters = std::make_tuple(i.coeffs[5]);
                  float c1 = 0.090554739064;
                  float c2 = -0.263085401898;
                  float c3 = 0;
                  float c4 = 0;
                  float c5 = 0;
                  auto matrix_parameters = std::make_tuple(c1,c2,c3,c4,c5);
                  rs2_distortion model = in.model;
                                    //print the parameters
                 // cout << "Resolution: " << resolution.first << "," << resolution.second << endl;
                 // cout << "Principle point: " << principal_point.first << "," << principal_point.second << endl;
                 // cout << "Focal length: " << focal_length.first << "," << focal_length.second << endl;
                 // cout << "Matrix parameters are: " << std::get<0>(matrix_parameters)<< endl;
                 // cout << "Matrix parameters are: " << std::get<1>(matrix_parameters)<< endl;
                 // cout << "Matrix parameters are: " << std::get<2>(matrix_parameters)<< endl;
                 // cout << "Matrix parameters are: " << std::get<3>(matrix_parameters)<< endl;
                 // cout << "Matrix parameters are: " << std::get<4>(matrix_parameters)<< endl;
                  //return 0;

// abbiamo centrop con i valori sull'immagine del centro del bounding box
// e due vettori in 3d dove poter inserire i valori reali
// facciamo la conversione con leggi geometriche
// vedi funzione definita da realsense che ho copiato  rs2_deproject_pixel_to_point


                  float x = (centrop[0] - principal_point.first) / focal_length.first;
                  float y = (centrop[1] - principal_point.second) / focal_length.second;
                      float r2  = x*x + y*y;
                      float f = 1 + std::get<0>(matrix_parameters)*r2 + std::get<1>(matrix_parameters)*r2*r2 + std::get<4>(matrix_parameters)*r2*r2*r2;
                      float ux = x*f + 2*std::get<2>(matrix_parameters)*x*y + std::get<3>(matrix_parameters)*(r2 + 2*x*x);
                      float uy = y*f + 2*std::get<3>(matrix_parameters)*x*y + std::get<2>(matrix_parameters)*(r2 + 2*y*y);
                      x = ux;
                      y = uy;

// il punto in 3D ha valori dati da depth*x, depth*y, depth
// m[0] � il valore medio della profondit�

                  centrov[0] = centerdepth * x;
                  centrov[1] = centerdepth * y;
                  centrov[2] = centerdepth;
                  float d;
                         d = m[0];
                  centrov1[0] = d * x;                           //NON NECESSARY POINT
                  centrov1[1] = d * y;                           //IT IS ONLY NEEDED TO CHECK DIFFERENCE FROM THE OTHER POINT
                  centrov1[2] = d;

                  cout << "THE PERSON IS CENTERED IN POINT: (" << centrop[0] << "," << centrop[1] << ")." << endl;
                  cout << "THE MEASURED DEPTH measured with get depth IS: " << centerdepth << "." << endl;
                  cout << "THE MEAN DEPTH (the one of the image) IS: " << m[0] << "." << endl;
                  cout << "WE CAN SEE THAT THE TWO DEPTH VALUE DON'T DIFFERE TOO MUCH SO WE ASSUME IT IS RIGHT."<< endl;
                  cout << "COORDINATES OF THE POINT IN THE WORD REFERENE FRAME ARE: (" << centrov[0] << "," << centrov[1] << "," << centrov[2] << ")." <<endl;
                  cout << "COORDINATES OF THE POINT IN THE WORD REFERENE FRAME ARE: (" << centrov1[0] << "," << centrov1[1] << "," << centrov1[2] << ")." <<endl;
                  cout << "WE REMIND THAT THE POSITIVE AXES POINT ON RIGHT, DOWN, INSIDE THE SCREEN."<< endl;
                  cout << "WE ARE LOOKING FOR THE POINT ON THE GROUND SO WE SHOULD ASSUME THE SECOND COORINATE EQUAL TO 0: (" << centrov[0] << "," << 0 << "," << centrov[2] << ")." << endl;
                  cout << "We should now save all points to make a path planning." << endl;

                  // il frame rate � di 30 fps quindi uno ogni 1 ogni 0.033 sec. una persona in media cammina a 6 km/h quindi 1.4m/sec per essere cautelativi 1.6m/sec
                  // ogni 0.033 sec si fanno 0.053m

                  //ho scelto mezzo secondo quindi mezzo metro di spostamento perch� la precisione delle misure non � elevata, ma comunque dell'ordine di grandezza di meno del mezzo metro
                  float dist = 0.8;

                  if(i < 15) //tanto per il primo mezzo secondo � l'unica persona vicina alla valigia e non si pu� sbagliare inoltre abbiamo delle caratteristiche
                  {
                      save[i][0] = centrov1[0];
                      save[i][1] = 0;
                      save[i][2] = centrov1[2];
                  }

                  Mat histogram1[i];
                  Mat histogram2[i];         // definiamo la variabile istogramma come array di valori
                  Mat histogram3[i];

                  Mat src, dst;              // creiamo le matrici necessarie
                  src = color_mat;           // carichiamo l'immagine da confrontare come object vedi def sopra come rettangolo bounding box cos� per� mi carica tutto il frame

                  Mat src1(src, Rect(xLeftBottom, yLeftBottom, xRightTop, yRightTop) );

                  vector<Mat> bgr_planes;
                  split( src1, bgr_planes );

                  int histSize = 256;
                  float range[] = { 0, 256 } ;
                  const float* histRange = { range };

                  bool uniform = true; bool accumulate = false;

                  Mat b_hist, g_hist, r_hist;

                  // Compute the histograms:
                  calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
                  calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
                  calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );


                  // Draw the histograms for B, G and R
                  int hist_w = 512; int hist_h = 400;
                  int bin_w = cvRound( (double) hist_w/histSize );

                  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

                  // Normalize the result to [ 0, histImage.rows ]
                  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
                  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
                  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

                  histogram1[i]= b_hist;
                  histogram2[i]= g_hist;
                  histogram3[i]= r_hist;

                  // Draw for each channel
                  for( int hs = 1; hs < histSize; hs++ )
                  {
                            line( histImage, Point( bin_w*(hs-1), hist_h - cvRound(b_hist.at<float>(hs-1)) ) ,
                                             Point( bin_w*(hs), hist_h - cvRound(b_hist.at<float>(hs)) ),
                                             Scalar( 255, 0, 0), 2, 8, 0  );
                            line( histImage, Point( bin_w*(hs-1), hist_h - cvRound(g_hist.at<float>(hs-1)) ) ,
                                             Point( bin_w*(hs), hist_h - cvRound(g_hist.at<float>(hs)) ),
                                             Scalar( 0, 255, 0), 2, 8, 0  );
                            line( histImage, Point( bin_w*(hs-1), hist_h - cvRound(r_hist.at<float>(hs-1)) ) ,
                                             Point( bin_w*(hs), hist_h - cvRound(r_hist.at<float>(hs)) ),
                                             Scalar( 0, 0, 255), 2, 8, 0  );
                  }
                  /// Display
                    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
                    imshow("calcHist Demo", histImage ); // tutta la parte dalla definizione di histogram 1,2,3 serve solo per l'immagine


                  float t = 0.1; //threshold da dare al cambiamento dell'istogramma bisogna capire come visualizzare come sigolo valore l'istogramma
                  // per questo valore ci vorrebbero dei try and attempt, pu� essere tranquillamente pi� grosso ma � un rischio che ritengo inutile
                  // praticamente vuol dire che tra 5 histogram consecutivi c'� una corrispondenza di almeno il 10 % su tutti i colori.

                       if(i>15)
                       {
                          if(abs(compareHist( histogram1[i], histogram1[i-5], 0 ))>t) // ovviamente se l'istogramma � dato da tre calori bisogna fare tre if su ogni valore dell'istogramma
                          {
                              if(abs(compareHist( histogram2[i], histogram2[i-5], 0 ))>t) // ovviamente se l'istogramma � dato da tre calori bisogna fare tre if su ogni valore dell'istogramma
                              {
                                  if(abs(compareHist( histogram3[i], histogram3[i-5], 0 ))>t) // ovviamente se l'istogramma � dato da tre calori bisogna fare tre if su ogni valore dell'istogramma
                                  {
                                      if(fabs(save[i][0]) < fabs(save[i-15][0]+ dist) || fabs(save[i][0]) > fabs(save[i-15][0]- dist) )
                                      {
                                          if(fabs(save[i][2]) < fabs((save[i-15][2]+ dist)) || fabs(save[i][2]) > fabs(save[i-15][2]- dist) )
                                          {
                                              save[i][0] = centrov1[0];
                                              save[i][1] = 0;
                                              save[i][2] = centrov1[2];
                                          }
                                      }
                                  }
                              }
                          }
                       }



                       // N.B.:
                       // ho usato centrov1 perch� ho valutare quale sia il valore pi� realistico
                       // centrov1 � con la media e sembra il pi� veritiero
                       // logica vorrebbe che il valore della profondit� del punto medio (CENTROV) sia pi� realistico, ma sembra essere meno affidabile
                       // LASCIO TUTTE LE RIGHE STAMPATE SOPRA COME CHECK DI QUALE PUNTO SIA IL MIGLIORE.
                       // IN SEGUITO POSSONO ESSERE TOLTE

                        cout << "Array con i punti salvati ogni volta dovrebbe essere pi� lungo di un elemento" << save << "." << endl;

               }
        }

        imshow(window_name, color_mat);                                                                      //show the rectangle with that image and color
        if (waitKey(1) >= 0) break;                                                                          //close function
    }

    return EXIT_SUCCESS;
}
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
