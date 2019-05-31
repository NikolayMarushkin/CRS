#include <omp.h>
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "Main.h"

using namespace std;

struct CRSMatrix
{
	int n; // ����� ����� � ������� 
	int m; // ����� �������� � ������� 
	int nz; // ����� ��������� ��������� � ����������� ������� 
	vector<double> val;   // ������ �������� ������� �� ������� 
	vector<int> colIndex; // ������ ������� �������� 
	vector<int> rowPtr;   // ������ �������� ������ ����� 
};

double MultVV(double* x, double* y, int n)
{
	double res = 0;
	for (int i = 0; i < n; i++)
	{
		res += x[i] * y[i];
	}
	return res;
}

void MultMatVec(CRSMatrix& A, double* x, double* result)
{
	//� = (U + D) * x + U^T * x  (U - ������� �����������, D - ���������
	for (int i = 0; i < A.n; i++)
	{
		result[i] = 0;
	}
	for (int i = 0; i < A.n; i++)
	{
		if (i < A.n - 1)
		{
			for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; j++)
			{
				// (U + D) * x
				result[i] += A.val[j] * x[A.colIndex[j]];
				// U^T * x
				if (A.colIndex[j] != i)
				{
					result[A.colIndex[j]] += A.val[j] * x[i];
				}
			}
		}
		else
		{
			for (int j = A.rowPtr[i]; j < A.nz; j++)
			{
				// (U + D) * x
				result[i] += A.val[j] * x[A.colIndex[j]];
				// U^T * x
				if (A.colIndex[j] != i)
				{
					result[A.colIndex[j]] += A.val[j] * x[i];
				}
			}
		}
	}
}


void SLE_Solver_CRS(CRSMatrix & A, double * b, double eps, int max_iter, double * x, int & count)
{
	for (int i = 0; i < A.n; i++) 
		x[i] = 1;

	double* Ax = new double[A.n];
	double* r = new double[A.n];
	double* p = new double[A.n];
	MultMatVec(A, x, Ax);
	for (int i = 0; i < A.n; i++)
	{
		r[i] = b[i] - Ax[i]; // ������� i-�� �����������
		p[i] = r[i];
	}
	// ���������� ��������� ����� ������� �������
	double StartNorm = sqrt(MultVV(b, b, A.n));
	double* Ap = new double[A.n];

	for (int k = 1; k <= max_iter; k++)
	{
		MultMatVec(A, p, Ap);
		double denominator = MultVV(Ap, p, A.n); //�����������
		if (denominator == 0) return;
		//Alphai ������� ������ ����������� ������� F �� ����������� pi
		double Alpha = MultVV(r, r, A.n) / denominator;

		double CurrentNorm = 0;
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < A.n; i++)
		{
			x[i] += Alpha * p[i];
			CurrentNorm += pow(Alpha * p[i], 2);
		}
		// ���������� ����� ������� ������� � ���������� ��������
		CurrentNorm = sqrt(CurrentNorm);
		//�������� ���������
		if (CurrentNorm / StartNorm < eps) return;

		double* rNext = new double[A.n]; //r + 1
		for (int i = 0; i < A.n; i++)
		{
			rNext[i] = r[i] - Alpha * Ap[i];
		}
		denominator = MultVV(r, r, A.n);
		if (denominator == 0) return;
		//Beta �����������, ��������������� ���������� ������� ������������� �����������
		double Beta = MultVV(rNext, rNext, A.n) / denominator;
		for (int i = 0; i < A.n; i++)
		{
			p[i] = rNext[i] + Beta * p[i];
		}
		delete[] r;
		r = rNext;
	}
}

int main()
{
	int n = 5;
	CRSMatrix A;
	A.m = n; A.n = n; A.nz = 8;
	double val[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
	vector<double> val1(val, val + 8);
	int col[] = { 0, 2, 4, 2, 4, 3, 3, 4 };
	vector<int> col1(col, col + 8);
	int row[] = { 0, 3, 5, 6, 7 };
	vector<int> row1(row, row + 5);
	A.val = val1; 
	A.colIndex = col1;
	A.rowPtr = row1;
	double x[] = { 1, 1, 1, 1, 1 };
	double* y = new double[A.n];
	//MultMatVec(A, x, y);
	MultMatVec(A, x, y);
	for (int i = 0; i < n; i++)
		cout << y[i] << endl;
	cout << endl;
	double* X = new double[n];
	int count;
	SLE_Solver_CRS(A, y, 0.0001, 10, X, count);
	system("Pause");
	return 0;
}
