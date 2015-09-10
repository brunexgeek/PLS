#ifndef PLS_HH
#define PLS_HH


#include <opencv2/opencv.hpp>


/**
 * Class to train and project PLS models.
 *
 * This implementation is based on the python PLS implementation by Avinash Kak (kak@purdue.edu).
 * Both implementations are based on the description of the algorithm by Herve Abdi in
 * the article "Partial Least Squares Regression and Projection on Latent Structure
 * Regression," Computational Statistics, 2010.
 */
class PartialLeastSquares
{
	private:
		cv::Mat B, mean0X, mean0Y;
		const int INTERNAL_TYPE = CV_64F;

		inline void display(
			const char *name,
			const cv::Mat &value );

	public:
		PartialLeastSquares( );
		PartialLeastSquares( const cv::Mat &B, const cv::Mat &meanX, const cv::Mat &meanY );
		PartialLeastSquares( const char *fileName );
		~PartialLeastSquares();

		void train(
			const cv::Mat &Xdata,
			const cv::Mat &Ydata,
			double epsilon = 0.0001 );

		const cv::Mat &getB() const;

		const cv::Mat &getMeanX() const;

		const cv::Mat &getMeanY() const;

		cv::Mat project(
			const cv::Mat &v ) const;

		void save( const char *fileName ) const;
};


#endif // PLS_HH
