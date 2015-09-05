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
		bool debug;

		inline void display(
			const char *name,
			const cv::Mat &value );

	public:
		PartialLeastSquares( bool debug = false );
		~PartialLeastSquares();

		void train(
			const cv::Mat &Xdata,
			const cv::Mat &Ydata,
			double epsilon = 0.0001 );

		inline const cv::Mat &getBeta() const;

		inline const cv::Mat &getMeanX() const;

		inline const cv::Mat &getMeanY() const;

		cv::Mat project(
			const cv::Mat &v ) const;
};


#endif // PLS_HH
