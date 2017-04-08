// Created on Tue Mar 28 21:04:37 2017
//
// @author: Ivan Koryakovskiy <i.koryakovskiy@gmail.com>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <cstring>
#include <omp.h>

template<class T>
bool safe_delete(T **obj)
{
    if (*obj)
    {
      delete *obj;
      *obj = NULL;
      return true;
    }
    else
      return false;
}

template<class T>
bool safe_delete_array(T **obj)
{
    if (*obj)
    {
      delete[] *obj;
      *obj = NULL;
      return true;
    }
    else
      return false;
}

class rbfBase
{
  public:
    rbfBase(char* name, const int *size, const int *dsize, int num, const double *cx, const double *cy, const double *cz, double sigma) :
      num_(num), sigma_(sigma), name_(name)
    {
      if (num_ < 1)
        return;

      std::cout << "Calling constructor of class " << name_ << std::endl;

      size_ = new int[3];
      for (int i = 0; i < 3; i++)
          size_[i] = size[i];

      q_ = new double[size_[0]*size_[1]*size_[2]];

      memset(cz_be_en_, num_, sizeof(cz_be_en_));

      cx_ = new double[num_];
      cy_ = new double[num_];
      cz_ = new double[num_];
      for (int i = 0; i < num_; i++)
      {
        cx_[i] = cx[i];
        cy_[i] = cy[i];
        int z = cz_[i] = cz[i];

        cz_be_en_[z][1] = i;     // last index
        if (cz_be_en_[z][0] > i)
          cz_be_en_[z][0] = i;   // first index
      }

      for (int i = 0; i < 3; i++)
      {
        dsize_[i] = dsize[i];
        //std::cout << cz_be_en_[i][0] << "  " << cz_be_en_[i][1] << " = " << dsize_[i] << std::endl;
      }
/*
      #pragma omp parallel for
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          std::cout << i << " " << j << " " << omp_get_thread_num() << std::endl;
*/
    }

    virtual double *evaluate(const double *f) = 0;
/*
      std::cout << "Internal size " << size_[0]*size_[1]*size_[2] << std::endl;
      for (int i = 0; i < dsize_[0]*dsize_[1]*dsize_[2]; i++)
        std::cout << f[i] << std::endl;

      int z = 0;
      double xx = 0.75, yy = 0.75;
      double q = 0;
      for (int i = cz_be_en_[z][0]; i <= cz_be_en_[z][1]; i++)
      {
        std::cout << cx_[i] << " " << cy_[i] << std::endl;

        double dist2 = pow(xx - cx_[i], 2) + pow(yy - cy_[i], 2);
        int dx = cx_[i]*dsize_[0] - 0.5;
        int dy = cy_[i]*dsize_[1] - 0.5;
        std::cout << dx << " " << dy << std::endl;
        int f_idx = round(dx + dy*dsize_[0] + z*dsize_[0]*dsize_[1]);
        std::cout << f_idx << " eval as " << f[f_idx] << std::endl;
        q += f[f_idx] * exp(- dist2 / (sigma_*sigma_));
      }
      std::cout << q << std::endl;
*/
    virtual void check_idxs(int idx, int didx)
    {
      if ( (idx < 0) || (idx >= size_[0]*size_[1]*size_[2]) )
      {
        std::cout << "Invalid index " << idx << "used to access array of size " <<  size_[0]*size_[1]*size_[2] << std::endl;
        exit(-1);
      }

      if ( (didx < 0) || (didx >= dsize_[0]*dsize_[1]*dsize_[2]) )
      {
        std::cout << "Invalid index " << didx << "used to access array of size " <<  dsize_[0]*dsize_[1]*dsize_[2] << std::endl;
        exit(-2);
      }
    }

    ~rbfBase()
    {
      std::cout << "Calling destructor of class " << name_ << std::endl;
      safe_delete_array(&q_);
      safe_delete_array(&cx_);
      safe_delete_array(&cy_);
      safe_delete_array(&cz_);
      safe_delete_array(&size_);
    }


  protected:
    double *q_;
    int num_;
    int *size_;
    std::string name_;
    double sigma_;
    double *cx_, *cy_, *cz_;
    int cz_be_en_[3][2];
    int dsize_[3];
};

class rbf : public rbfBase
{
  public:
    rbf(char *name, const int *size, const int *dsize, int num, const double *cx, const double *cy, const double *cz, double sigma) :
      rbfBase(name, size, dsize, num, cx, cy, cz, sigma)
    {}

    virtual double *evaluate(const double *f)
    {
      memset(q_, 0, sizeof(double)*size_[0]*size_[1]*size_[2]);

      for (int z = 0; z < size_[2]; z++)
      {
        #pragma omp parallel for collapse(2)
        for (int x = 0; x < size_[0]; x++)
        {
          for (int y = 0; y < size_[1]; y++)
          {
            int idx = x + y*size_[0] + z*size_[0]*size_[1];
            double xx = (x+0.5)/size_[0];
            double yy = (y+0.5)/size_[1];

            for (int i = cz_be_en_[z][0]; i <= cz_be_en_[z][1]; i++)
            {
              double dist2 = pow(xx - cx_[i], 2) + pow(yy - cy_[i], 2);
              int dx = cx_[i]*dsize_[0] - 0.5;
              int dy = cy_[i]*dsize_[1] - 0.5;
              int didx = round(dx + dy*dsize_[0] + z*dsize_[0]*dsize_[1]);
              check_idxs(idx, didx);
              q_[idx] += f[didx] * exp(- dist2 / (sigma_*sigma_));
            }
            //std::cout << q_[idx] << std::endl;
          }
        }
      }
      return q_;
    }
};

class nrbf : public rbfBase
{
  public:
    nrbf(char* name, const int *size, const int *dsize, int num, const double *cx, const double *cy, const double *cz, double sigma) :
      rbfBase(name, size, dsize, num, cx, cy, cz, sigma)
    {}

    virtual double *evaluate(const double *f)
    {
      memset(q_, 0, sizeof(double)*size_[0]*size_[1]*size_[2]);

      for (int z = 0; z < size_[2]; z++)
      {
        #pragma omp parallel for collapse(2)
        for (int x = 0; x < size_[0]; x++)
        {
          for (int y = 0; y < size_[1]; y++)
          {
            int idx = x + y*size_[0] + z*size_[0]*size_[1];
            double xx = (x+0.5)/size_[0];
            double yy = (y+0.5)/size_[1];
            double sum = 0;

            for (int i = cz_be_en_[z][0]; i <= cz_be_en_[z][1]; i++)
            {
              double dist2 = pow(xx - cx_[i], 2) + pow(yy - cy_[i], 2);
              int dx = cx_[i]*dsize_[0] - 0.5;
              int dy = cy_[i]*dsize_[1] - 0.5;
              int didx = round(dx + dy*dsize_[0] + z*dsize_[0]*dsize_[1]);
              check_idxs(idx, didx);
              double weight = exp(- dist2 / (sigma_*sigma_));
              q_[idx] += f[didx] * weight;
              sum += weight;
            }
            q_[idx] /= sum;
            //std::cout << q_[idx] << std::endl;
          }
        }
      }
      return q_;
    }
};
/*
class tst : public rbfBase
{
  public:
    tst() :
      rbfBase(NULL, NULL, 0, NULL, NULL, NULL, 0)
    {}

    virtual double *evaluate(const double *f)
    {
      return NULL;
    }
};
*/
extern "C"
{
  // Classical RBF
  rbf* rbf_new(char* name, const int *size, const int *dsize, int num,
                const double *cx, const double *cy, const double *cz, double sigma)
  {
    return new rbf(name, size, dsize, num, cx, cy, cz, sigma);
  }

  // Normalized RBF
  nrbf* nrbf_new(char* name, const int *size, const int *dsize, int num,
                const double *cx, const double *cy, const double *cz, double sigma)
  {
    return new nrbf(name, size, dsize, num, cx, cy, cz, sigma);
  }
/*
  tst* tst_new()
  {
    return new tst();
  }
*/

  double *rbf_evaluate(rbfBase* r, const double *f){ return r->evaluate(f); }

  void clear(rbfBase* r){ delete r; }
}