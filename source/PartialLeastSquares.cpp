#include <PLS/PartialLeastSquares.hh>
#include <math.h>
#include <cstdio>



PartialLeastSquares::PartialLeastSquares( )
{
	// nothing to do
}


PartialLeastSquares::PartialLeastSquares(
	const cv::Mat &B,
	const cv::Mat &meanX,
	const cv::Mat &meanY ) : B(B), mean0X(meanX), mean0Y(meanY)
{
	// nothing to do
}


PartialLeastSquares::PartialLeastSquares( const char *fileName )
{
	cv::FileStorage fs(fileName, cv::FileStorage::READ);
	fs["B"] >> B;
	fs["meanX"] >> mean0X;
	fs["meanY"] >> mean0Y;
	fs.release();
}


PartialLeastSquares::~PartialLeastSquares()
{
	// nothing to do
}


void PartialLeastSquares::display(
	const char *name,
	const cv::Mat &value )
{
	std::cout << std::endl << name << std::endl << value << std::endl;
}


void PartialLeastSquares::train(
	const cv::Mat &Xdata,
	const cv::Mat &Ydata,
	double epsilon )
{
	if (Xdata.type() != Ydata.type())
		return;

	cv::Mat X, Y;

	Xdata.convertTo(X, INTERNAL_TYPE);
	Ydata.convertTo(Y, INTERNAL_TYPE);

	cv::reduce(X, mean0X, 0, CV_REDUCE_AVG, INTERNAL_TYPE);
	cv::reduce(Y, mean0Y, 0, CV_REDUCE_AVG, INTERNAL_TYPE);

    for (int i = 0; i < X.rows; ++i)
		X.row(i) = X.row(i) - mean0X;

    for (int i = 0; i < Y.rows; ++i)
		Y.row(i) = Y.row(i) - mean0Y;

#if (0)
	display("Column-wise mean for X:", mean0X);
	display("Zero-mean version of X:", X);
	display("Column-wise mean for Y:", mean0Y);
	display("Zero-mean version of Y:", Y);
#endif

    cv::Mat T, U, W, C, P, Q, Bdiag, t, w, u, c, p, q, b;
	u = cv::Mat(X.rows, 1, INTERNAL_TYPE);
	cv::randu(u, cv::Scalar::all(0), cv::Scalar::all(1));

#if (0)
	display("The initial random guess for u:", u);
#endif

	int i = 0;
	while (1)
	{
		int j = 0;
		while (1)
		{
			w = X.t() * u;
			w = w / cv::norm(w);
			t = X * w;
			t = t / cv::norm(t);
			c = Y.t() * t;
			c = c / cv::norm(c);
			cv::Mat u_old;
			u.copyTo(u_old);
			u = Y * c;
			double error = cv::norm(u - u_old);
			if (error < epsilon)
			{
				/*if (debug)
					std::cout << "Number of iterations for the " << i << "th latent vector: " << j+1 << std::endl;*/
				break;
			}
			j += 1;
		}
		b = t.t() * u;
		assert(b.cols == 1 && b.rows == 1);
#if (0)
		if (T.cols == 0)
			t.copyTo(T);
		else
			cv::hconcat(T, t, T);

		if (U.cols == 0)
			u.copyTo(U);
		else
			cv::hconcat(U, u, U);

		if (W.cols == 0)
			w.copyTo(W);
		else
			cv::hconcat(W, w, W);
#endif
		if (C.cols == 0)
			c.copyTo(C);
		else
			cv::hconcat(C, c, C);

		double temp = cv::norm(t);
		p = X.t() * t / (temp * temp);
#if (0)
		temp = cv::norm(u);
		q = Y.t() * u / (temp * temp);
#endif
		if (P.cols == 0)
			p.copyTo(P);
		else
			cv::hconcat(P, p, P);
#if (0)
		if (Q.cols == 0)
			q.copyTo(Q);
		else
			cv::hconcat(Q, q, Q);
#endif
		if (Bdiag.cols == 0)
			b.copyTo(Bdiag);
		else
			cv::hconcat(Bdiag, b, Bdiag);

		X = X - t * p.t();
		Y = Y - b.at<double>(0,0) * t * c.t();
		i += 1;
		if (cv::norm(X) < 0.001) break;
	}
	Bdiag = cv::Mat::diag(Bdiag);
	B = P.t().inv(cv::DECOMP_SVD);
	B = B * Bdiag;
	B = B * C.t();

#if (0)
	display("The T matrix:", T);
	display("The U matrix:", U);
	display("The W matrix:", W);
	display("The C matrix:", C);
	display("The P matrix:", P);
	display("The b matrix:", Bdiag);
	display("The diagonal matrix B of b values:", Bdiag);
	display("The matrix B of regression coefficients:", B);
#endif
}


const cv::Mat &PartialLeastSquares::getB() const
{
	return this->B;
}


const cv::Mat &PartialLeastSquares::getMeanX() const
{
	return this->mean0X;
}


const cv::Mat &PartialLeastSquares::getMeanY() const
{
	return this->mean0Y;
}


cv::Mat PartialLeastSquares::project(
	const cv::Mat &v ) const
{
	cv::Mat temp;
	cv::Mat result = cv::Mat::zeros(v.rows, B.cols, INTERNAL_TYPE);
	v.convertTo(temp, INTERNAL_TYPE);

	for (int i = 0; i < temp.rows; ++i)
	{
		// subtract the training X mean
		temp.row(i) -= mean0X;
		// predict the Y matrix
		result.row(i) = temp.row(i) * B;
		// add the training X mean
		result.row(i) += mean0Y;
	}

	return result;
}


void PartialLeastSquares::save( const char *fileName ) const
{
	cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
	fs << "B" << B;
	fs << "meanX" << mean0X;
	fs << "meanY" << mean0Y;
	fs.release();
}

