#include <iostream>
#include <iomanip>
#include <math.h>
#include <cstring>
#include <omp.h>

class rbf
{
  public:
    rbf(const int *size, const int *dsize, int num, 
          const double *cx, const double *cy, const double *cz, double sigma) :
        num_(num), sigma_(sigma)
    {
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
       
    double *evaluate(const double *f)
    {
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
              int f_idx = round(dx + dy*dsize_[0] + z*dsize_[0]*dsize_[1]);
              q_[idx] += f[f_idx] * exp(- dist2 / (sigma_*sigma_));
            }
          }
        }
      }

      return q_;
    }
  
  private:
    double *q_;
    int num_;
    int *size_; 
    double *cx_, *cy_, *cz_;
    double sigma_;
    int cz_be_en_[3][2];
    int dsize_[3];
};


extern "C" 
{
  rbf* rbf_new(const int *size, const int *dsize, int num, 
                const double *cx, const double *cy, const double *cz, double sigma)
  {
    return new rbf(size, dsize, num, cx, cy, cz, sigma); 
  }
  double *rbf_evaluate(rbf* r, const double *f){ return r->evaluate(f); }
}