#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <algorithm>
#include <cmath>
using namespace cv;
using namespace std;

int OtsuThreshold(const Mat& img);
int main( int argc, char** argv)
{
    Mat img_original, img_gray;
    //이미지파일을 로드하여 image에 저장
    img_original = imread( "test3.jpg", IMREAD_COLOR );
    if (img_original.empty()){
        printf("Could not open or find the image\n");
        return -1;
    }
    //그레이스케일 이미지로 변환
    cvtColor( img_original, img_gray, COLOR_BGR2GRAY);
    int threshold = OtsuThreshold(img_gray);
    printf("threshold: %d\n", threshold);
    //그레이스케일 이미지가 아닐 때
    if(threshold == -1){
      printf("image is not gray scale\n");
      return -1;
    }
    for(int x = 0;x < img_gray.rows;x++){
      for(int y = 0;y < img_gray.cols;y++){
        img_gray.at<uchar>(x, y) = (img_gray.at<uchar>(x, y) >= threshold ? 255 : 0);
      }
    }
    //윈도우 생성
    namedWindow( "original image", WINDOW_AUTOSIZE);
    namedWindow( "gray image", WINDOW_AUTOSIZE);
    //윈도우에 출력
    imshow( "original image", img_original );
    imshow( "gray image", img_gray);
    //키보드 입력이 될때까지 대기
    waitKey(0);

    //디스크에 저장
    imwrite("test_gray.jpg", img_gray );
    return 0;
}
//OtsuThreshold 경계를 찾아서 idx값 리턴
int OtsuThreshold(const Mat& img){
  //히스토그램 , 누적합 히스토그램 (구간합 연산 빠르게 하기 위함)
  double hist[256] = {0}, accum_hist[256] = {0}, accum_iPi[256] = {0};

  long uT = 0; //0부터 256까지 i*Pi의 합
  //흑백영상이 아닐 때 return -1;
  if(img.channels() != 1)  return -1;
  //흑백영상의 값을 hist에 저장(histogram 만들기)
  for(int x = 0;x < img.rows;x++){
    for(int y = 0;y < img.cols;y++){
      hist[(img.at<uchar>(x, y))] += 1;
    }
  }
  //정규화 하기
  int N = img.rows * img.cols; //N: total# of pixels
  hist[0] /= N;
  accum_hist[0] = hist[0]; //누적합
  for(int i = 1;i < 256;i++){
    hist[i] /= N;
    accum_hist[i] = accum_hist[i - 1] + hist[i];
    accum_iPi[i] = accum_iPi[i - 1] + i * hist[i];
  }

  double sigMin = 1 << 29;
  int arg = -1;
  for(int i = 0;i < 256;i++){
    //w0: 0~i까지 합, w1: i + 1 ~ 255까지 합
    double w0 = accum_hist[i], w1 = accum_hist[255] - accum_hist[i];
    double u0 = accum_iPi[i] / w0, u1 = 0;
    if(w1 != 0.)
      u1 = (accum_iPi[255] - accum_iPi[i]) / w1;
    double sigB = w0 * pow(u0 - accum_iPi[255], 2) + w1 * pow(u1 - accum_iPi[255], 2);
    if(sigMin > sigB)
      sigMin = sigB, arg = i;
    //else if(sigMin == sigB) mulCount++;
  }
  //if(mulCount > 1)  arg = arg + (mulCount - 1) / 2;
  return arg;
}
