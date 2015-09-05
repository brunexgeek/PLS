#include <PLS/PartialLeastSquares.hh>
#include <math.h>



PartialLeastSquares::PartialLeastSquares( bool debug ) : debug(debug)
{
	// nothing to do
}


PartialLeastSquares::~PartialLeastSquares()
{
	// nothing to do
}


void PartialLeastSquares::display(
	const char *name,
	const cv::Mat &value )
{
	if (debug) std::cout << std::endl << name << std::endl << value << std::endl;
}


void PartialLeastSquares::train(
	const cv::Mat &Xdata,
	const cv::Mat &Ydata,
	double epsilon )
{
	cv::Mat X, Y;
	int i ;

	B = cv::Mat();

	Xdata.copyTo(X);
	Ydata.copyTo(Y);

	display("Column-wise mean for X:", Xdata);
	cv::reduce(X, mean0X, 0, CV_REDUCE_AVG, CV_32F);
	display("Column-wise mean for X:", mean0X);

    for (i = 0; i < X.rows; ++i)
		X.row(i) = X.row(i) - mean0X;
    display("Zero-mean version of X:", X);

	cv::reduce(Y, mean0Y, 0, CV_REDUCE_AVG, CV_32F);
	display("Column-wise mean for Y:", mean0Y);

    for (i = 0; i < Y.rows; ++i)
		Y.row(i) = Y.row(i) - mean0Y;
    display("Zero-mean version of Y:", Y);

    cv::Mat T, U, W, C, P, Q, Bdiag, t, w, u, c, p, q, b;

	//float values[1][5] = { { 0.1, 0.2, 0.3, 0.4, 0.5 } };
	u = cv::Mat(X.rows, 1, CV_32F/*, values*/);
	cv::randu(u, cv::Scalar::all(0), cv::Scalar::all(1));

	display("The initial random guess for u:", u);

	i = 0;
	int bi = 0;
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
				if (debug)
					std::cout << "Number of iterations for the " << i << "th latent vector: " << j+1 << std::endl;
				break;
			}
			j += 1;
		}
		b = t.t() * u;
		bi = b.at<float>(0,0);

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

		if (C.cols == 0)
			c.copyTo(C);
		else
			cv::hconcat(C, c, C);

		float temp = cv::norm(t);
		p = X.t() * t / (temp * temp);
		temp = cv::norm(u);
		q = Y.t() * u / (temp * temp);

		if (P.cols == 0)
			p.copyTo(P);
		else
			cv::hconcat(P, p, P);

		if (Q.cols == 0)
			q.copyTo(Q);
		else
			cv::hconcat(Q, q, Q);

		if (Bdiag.cols == 0)
			b.copyTo(Bdiag);
		else
			cv::hconcat(Bdiag, b, Bdiag);

		X = X - t * p.t();
		Y = Y - bi * t * c.t();
		i += 1;
		if (cv::norm(X) < 0.001) break;
	}

	display("The T matrix:", T);
	display("The U matrix:", U);
	display("The W matrix:", W);
	display("The C matrix:", C);
	display("The P matrix:", P);
	display("The b matrix:", Bdiag);
	//display("The final deflated X matrix:", X);
	//display("The final deflated Y matrix:", Y);

	B = cv::Mat::diag(Bdiag);
	display("The diagonal matrix B of b values:", B);
	cv::Mat K = P.t().inv(cv::DECOMP_SVD);
	K = K * B;
	B = K * C.t();
	display("The matrix B of regression coefficients:", B);
}


const cv::Mat &PartialLeastSquares::getBeta()
{
	return this->B;
}


const cv::Mat &PartialLeastSquares::getMeanX()
{
	return this->mean0X;
}


const cv::Mat &PartialLeastSquares::getMeanY()
{
	return this->mean0Y;
}


cv::Mat PartialLeastSquares::project(
	const cv::Mat &v )
{
	cv::Mat temp;

	// subtract the training X mean
	v.copyTo(temp);
	for (int i = 0; i < temp.rows; ++i)
		temp.row(i) -= mean0X;
	// predict the Y matrix
	temp *= B;
	// add the training X mean
	for (int i = 0; i < temp.rows; ++i)
		temp.row(i) += mean0Y;

	return temp;
}
