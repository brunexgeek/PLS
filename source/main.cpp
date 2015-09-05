#include <opencv2/opencv.hpp>
#include <PLS/PartialLeastSquares.hh>
#include <iostream>

using namespace cv;
using namespace std;


void main_display( const char *name, const Mat &value )
{
	std::cout << std::endl << name << std::endl << value << std::endl;
}


int main( int argc, char **argv )
{
	if (argc < 3)
	{
		std::cout << "Usage: pls <X data file> <Y data file>" << std::endl;
		return 1;
	}

	Mat X,Y;
	CvMLData csv;

	// load X data from CSV
	csv.read_csv(argv[1]);
	const CvMat *Xt = csv.get_values();
	X = Mat(Xt, true);
	main_display("X", X);

	// load Y data from CSV
	csv.read_csv(argv[2]);
	const CvMat *Yt = csv.get_values();
	Y = Mat(Yt, true);
	main_display("Y", Y);

	PartialLeastSquares pls;
	// train the PLS model
	pls.train(X, Y);
	// project the original X data
	cv::Mat result = pls.project(X);
	// show the projected Y matrix (compare with original Y)
	main_display("Projection", result);

	return 0;
}
