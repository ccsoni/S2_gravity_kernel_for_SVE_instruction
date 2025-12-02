#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include "MT.h"
#include "sve_macro.h"

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#ifdef __PROFILING__
#include "fj_tool/fapp.h"
#endif // __PROFILING__

#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))

#define NPIPE (16)
#define NIMAX (1024)
#define NJMAX (32768)

typedef float REAL;

struct iptcl {
  float xpos[NIMAX];
  float ypos[NIMAX];
  float zpos[NIMAX];
  float eps[NIMAX];
};

struct jptcl {
  float xpos, ypos, zpos, mass;
};

struct fodata {
  float xacc[NIMAX];
  float yacc[NIMAX];
  float zacc[NIMAX];
};

static inline REAL gfactor_acc(REAL rad, REAL eps_pm) 
{
  REAL R;
  REAL g;
  REAL S;

  R = 2.0*rad/eps_pm;
  R = (R > 2.0) ? 2.0 : R;
  S = R-1.0;
  S = (S > 0.0) ? S : 0.0;

  g = 1.0 + CUBE(R)*(-1.6+SQR(R)*(1.6+R*(-0.5+R*(0.15*R-12.0/35.0))))
    -CUBE(S)*CUBE(S)*(3.0/35.0+R*(18.0/35.0+0.2*R));

  if(R>2.0) g=0.0;

  return g;
}

static inline svfloat32_t rsqrt_acc(svbool_t pg, svfloat32_t _x)
{
  svfloat32_t _res = svrsqrte(_x);
  _res = svmul_x(pg, svrsqrts(svmul_x(pg, _res, _res), _x), _res);

  return _res;
}

static inline svfloat32_t gfactor_acc_sve(svbool_t pg, svfloat32_t _rad, float32_t sft_pm_inv)
{
  const float32_t zero = 0.0;
  const float32_t one = 1.0;
  const float32_t two = 2.0;

  const float32_t coeff0 = 0.15;
  const float32_t coeff1 = 12.0/35.0;
  const float32_t coeff2 = -0.5;
  const float32_t coeff3 = 1.6;
  const float32_t coeff4 = 0.2;
  const float32_t coeff5 = 18.0/35.0;
  const float32_t coeff6 = 3.0/35.0;

  svfloat32_t _zero = svdup_f32(zero);
  svfloat32_t _one  = svdup_f32(one);
  svfloat32_t _two  = svdup_f32(two); 

  svfloat32_t _g, _h, _R, _RR, _R3, _S;
  
  //  _R = svdiv_x(pg, svmul_x(pg, _rad, two), sft_pm);
  _R = svmul_x(pg, svmul_x(pg, _rad, _two), sft_pm_inv);
  _RR = _R;
  _R = svmin_x(pg, _R, _two);

  _R3 = svmul_x(pg, svmul_x(pg, _R, _R), _R);

  _S = svmax_x(pg, svsub_x(pg, _R, _one), zero);
  
  _S = svmul_x(pg, svmul_x(pg, _S, _S), _S); // S^3
  _S = svmul_x(pg, _S, _S); // S^6
  
  _g = svnmsb_x(pg, svdup_f32(coeff0), _R, coeff1);
  _g = svmad_x(pg, _g, _R, coeff2);
  _g = svmad_x(pg, _g, _R, coeff3);
  _g = svmul_x(pg, _g, _R);
  _g = svnmsb_x(pg, _g, _R, coeff3);
  _g = svmul_x(pg, _g, _R3);
  //_g = svmul_x(pg, _g, _R);
  //_g = svmul_x(pg, _g, _R);
  //_g = svmul_x(pg, _g, _R);
  
  _h = svmad_x(pg, svdup_f32(coeff4), _R, coeff5);
  _h = svmad_x(pg, _h, _R, coeff6);
  _h = svmul_x(pg, _h, _S);

  _g = svsub_x(pg, _g, _h);
  _g = svadd_x(pg, _g, one);

  _g = svsel(svcmpgt(pg, _RR, 2.0), _zero, _g);

  return _g;
}

void grav_S2_with_sve(struct iptcl *ip, struct fodata *result, int ni,
		      struct jptcl *jp, int nj, float sft_pm)
{
  static float dbg[16];
  static float dbg2[16];

  uint64_t i = 0;
  svbool_t pg_i = svwhilelt_b32(i, ni);
  svbool_t pg_j = svwhilelt_b32(0, 4);  
  
  float sft_pm_inv = 1.0/sft_pm;

  do {
    svfloat32_t _xacc = svdup_f32(0.0);
    svfloat32_t _yacc = svdup_f32(0.0);
    svfloat32_t _zacc = svdup_f32(0.0);
    
    svfloat32_t _xi = svld1(pg_i, ip->xpos+i);
    svfloat32_t _yi = svld1(pg_i, ip->ypos+i);
    svfloat32_t _zi = svld1(pg_i, ip->zpos+i);
    svfloat32_t _eps2 = svld1(pg_i, ip->eps+i);

    _eps2 = svmul_x(pg_i, _eps2, _eps2);
    
    for(int j=0;j<nj;j++) {
      svfloat32_t _mj = svld1(pg_j, (float32_t *)(jp+j));
      
      svfloat32_t _xj = svdup_lane(_mj, 0);
      svfloat32_t _yj = svdup_lane(_mj, 1);
      svfloat32_t _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);

      svfloat32_t _dx = svsub_x(pg_i, _xj, _xi);
      svfloat32_t _dy = svsub_x(pg_i, _yj, _yi);
      svfloat32_t _dz = svsub_x(pg_i, _zj, _zi);

      svfloat32_t _rsq = svmad_x(pg_i, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg_i, _dy, _dy, _rsq);
      _rsq = svmad_x(pg_i, _dz, _dz, _rsq);

#if __APPROX_RCP_RSQRT__
      svfloat32_t _rad = svmul_x(pg_i, svrsqrte(_rsq), _rsq);
#else
      svfloat32_t _rad = svmul_x(pg_i, rsqrt_acc(pg_i, _rsq), _rsq);
#endif
      svfloat32_t _gfact = gfactor_acc_sve(pg_i, _rad, sft_pm_inv);

#if __APPROX_RCP_RSQRT__
      svfloat32_t _mrinv3 = svrsqrte(svadd_x(pg_i, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
      svfloat32_t _mrinv3 = rsqrt_acc(pg_i, svadd_x(pg_i, _rsq, _eps2));
#endif
      _mrinv3 = svmul_x(pg_i, svmul_x(pg_i, _mrinv3, _mrinv3), _mrinv3);
      _mrinv3 = svmul_x(pg_i, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg_i, _gfact, _mrinv3);

      _xacc = svmad_x(pg_i, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg_i, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg_i, _mrinv3, _dz, _zacc);

    }

    svst1(pg_i, result->xacc+i, _xacc);
    svst1(pg_i, result->yacc+i, _yacc);
    svst1(pg_i, result->zacc+i, _zacc);

    i += svcntw();
    pg_i = svwhilelt_b32(i, ni);
  }while(svptest_any(svptrue_b32(),pg_i));
}

void grav_S2_with_sve_unroll16(struct iptcl *ip, struct fodata *result, int ni,     struct jptcl *jp, int nj, float sft_pm)
{
  static float dbg[16];
  static float dbg2[16];

  uint64_t i = 0;
  svbool_t pg = svwhilelt_b32(i, ni);
  svbool_t pg_j = svptrue_b32();

  float sft_pm_inv = 1.0/sft_pm;

  int nj16, mod_nj16;
  mod_nj16 = nj%16;
  nj16 = nj - mod_nj16;

  do {
    svfloat32_t _xacc = svdup_f32(0.0f);
    svfloat32_t _yacc = svdup_f32(0.0f);
    svfloat32_t _zacc = svdup_f32(0.0f);

    svfloat32_t _xi = svld1(pg, ip->xpos+i);
    svfloat32_t _yi = svld1(pg, ip->ypos+i);
    svfloat32_t _zi = svld1(pg, ip->zpos+i);
    svfloat32_t _eps2 = svld1(pg, ip->eps+i);

    _eps2 = svmul_x(pg, _eps2, _eps2);

    svfloat32_t _jp;
    svfloat32_t _mj, _xj, _yj, _zj;
    svfloat32_t _dx, _dy, _dz;
    svfloat32_t _rsq, _rad, _gfact, _mrinv3;

    for(int j=0;j<nj16;j+=16) {
      // 0
      _jp = svld1(pg_j, (float32_t *)(jp+j));
      _xj = svdup_lane(_jp, 0);
      _yj = svdup_lane(_jp, 1);
      _zj = svdup_lane(_jp, 2);

      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);

      _mj = svdup_lane(_jp, 3);      

      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
      _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
      _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
      _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
      _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

         // 1
         _xj = svdup_lane(_jp, 4);
         _yj = svdup_lane(_jp, 5);
         _zj = svdup_lane(_jp, 6);

      _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

         _mj = svdup_lane(_jp, 7);
      
      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

         _dx = svsub_x(pg, _xj, _xi);
         _dy = svsub_x(pg, _yj, _yi);
         _dz = svsub_x(pg, _zj, _zi);

         _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
         _rsq = svmad_x(pg, _dy, _dy, _rsq);
         _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
         _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
         _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
         _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
         _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
         _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

      // 2
      _xj = svdup_lane(_jp, 8);
      _yj = svdup_lane(_jp, 9);
      _zj = svdup_lane(_jp, 10);

         _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
         _mrinv3 = svmul_x(pg, _mj, _mrinv3);
         _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _mj = svdup_lane(_jp, 11);

         _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
         _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
         _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);

      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
      _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
      _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
      _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
      _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

         // 3
         _xj = svdup_lane(_jp, 12);
         _yj = svdup_lane(_jp, 13);
         _zj = svdup_lane(_jp, 14);

      _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

         _mj = svdup_lane(_jp, 15);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);      
      
         _dx = svsub_x(pg, _xj, _xi);
         _dy = svsub_x(pg, _yj, _yi);
         _dz = svsub_x(pg, _zj, _zi);

         _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
         _rsq = svmad_x(pg, _dy, _dy, _rsq);
         _rsq = svmad_x(pg, _dz, _dz, _rsq);

      // 4
      _jp = svld1(pg_j, (float32_t *)(jp+j+4));

#ifdef __APPROX_RCP_RSQRT__
         _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
         _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
         _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
         _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
         _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

      _xj = svdup_lane(_jp, 0);
      _yj = svdup_lane(_jp, 1);
      _zj = svdup_lane(_jp, 2);

         _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
         _mrinv3 = svmul_x(pg, _mj, _mrinv3);
         _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _mj = svdup_lane(_jp, 3);

         _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
         _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
         _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);

      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
      _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
      _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
      _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
      _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

      // 5
         _xj = svdup_lane(_jp, 4);
         _yj = svdup_lane(_jp, 5);
         _zj = svdup_lane(_jp, 6);

      _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

         _mj = svdup_lane(_jp, 7);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

         _dx = svsub_x(pg, _xj, _xi);
         _dy = svsub_x(pg, _yj, _yi);
         _dz = svsub_x(pg, _zj, _zi);

         _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
         _rsq = svmad_x(pg, _dy, _dy, _rsq);
         _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
         _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
         _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
         _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
         _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
         _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

      // 6
      _xj = svdup_lane(_jp, 8);
      _yj = svdup_lane(_jp, 9);
      _zj = svdup_lane(_jp, 10);

         _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
         _mrinv3 = svmul_x(pg, _mj, _mrinv3);
         _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _mj = svdup_lane(_jp, 11);

         _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
         _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
         _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);

      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
      _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
      _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
      _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
      _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

         // 7
         _xj = svdup_lane(_jp, 12);
         _yj = svdup_lane(_jp, 13);
         _zj = svdup_lane(_jp, 14);

      _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

         _mj = svdup_lane(_jp, 15);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

         _dx = svsub_x(pg, _xj, _xi);
         _dy = svsub_x(pg, _yj, _yi);
         _dz = svsub_x(pg, _zj, _zi);

         _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
         _rsq = svmad_x(pg, _dy, _dy, _rsq);
         _rsq = svmad_x(pg, _dz, _dz, _rsq);

      // 8
      _jp = svld1(pg_j, (float32_t *)(jp+j+8));

#ifdef __APPROX_RCP_RSQRT__
         _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
         _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
         _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
         _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
         _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

      _xj = svdup_lane(_jp, 0);
      _yj = svdup_lane(_jp, 1);
      _zj = svdup_lane(_jp, 2);

         _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
         _mrinv3 = svmul_x(pg, _mj, _mrinv3);
         _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _mj = svdup_lane(_jp, 3);

         _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
         _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
         _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);

      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
      _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
      _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
      _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
      _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

         // 9
         _xj = svdup_lane(_jp, 4);
         _yj = svdup_lane(_jp, 5);
         _zj = svdup_lane(_jp, 6);

      _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

	 _mj = svdup_lane(_jp, 7);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

         _dx = svsub_x(pg, _xj, _xi);
         _dy = svsub_x(pg, _yj, _yi);
         _dz = svsub_x(pg, _zj, _zi);

         _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
	 _rsq = svmad_x(pg, _dy, _dy, _rsq);
	 _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
	 _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
	 _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
	 _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
	 _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
	 _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

      // 10
      _xj = svdup_lane(_jp, 8);
      _yj = svdup_lane(_jp, 9);
      _zj = svdup_lane(_jp, 10);

	 _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
	 _mrinv3 = svmul_x(pg, _mj, _mrinv3);
	 _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _mj = svdup_lane(_jp, 11);

	 _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
	 _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
	 _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);

      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
      _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
      _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
      _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
      _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

         // 11
	 _xj = svdup_lane(_jp, 12);
	 _yj = svdup_lane(_jp, 13);
	 _zj = svdup_lane(_jp, 14);

      _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

	 _mj = svdup_lane(_jp, 15);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

         _dx = svsub_x(pg, _xj, _xi);
         _dy = svsub_x(pg, _yj, _yi);
         _dz = svsub_x(pg, _zj, _zi);

	 _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
	 _rsq = svmad_x(pg, _dy, _dy, _rsq);
	 _rsq = svmad_x(pg, _dz, _dz, _rsq);
	 
      // 12
      _jp = svld1(pg_j, (float32_t *)(jp+j+12));

#ifdef __APPROX_RCP_RSQRT__
	 _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
	 _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
	 _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
	 _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
	 _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

      _xj = svdup_lane(_jp, 0);
      _yj = svdup_lane(_jp, 1);
      _zj = svdup_lane(_jp, 2);

	 _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
	 _mrinv3 = svmul_x(pg, _mj, _mrinv3);
	 _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _mj = svdup_lane(_jp, 3);

	 _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
	 _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
	 _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);

      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
      _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
      _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
      _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
      _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

         // 13
	 _xj = svdup_lane(_jp, 4);
	 _yj = svdup_lane(_jp, 5);
	 _zj = svdup_lane(_jp, 6);
      
      _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

	 _mj = svdup_lane(_jp, 7);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

         _dx = svsub_x(pg, _xj, _xi);
	 _dy = svsub_x(pg, _yj, _yi);
	 _dz = svsub_x(pg, _zj, _zi);
	 
	 _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
	 _rsq = svmad_x(pg, _dy, _dy, _rsq);
	 _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
	 _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
	 _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
	 _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
	 _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
	 _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

      // 14
      _xj = svdup_lane(_jp, 8);
      _yj = svdup_lane(_jp, 9);
      _zj = svdup_lane(_jp, 10);

	 _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
	 _mrinv3 = svmul_x(pg, _mj, _mrinv3);
	 _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _mj = svdup_lane(_jp, 11);

	 _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
	 _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
	 _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);

      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
      _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
      _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
      _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
      _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

         // 15
	 _xj = svdup_lane(_jp, 12);
	 _yj = svdup_lane(_jp, 13);
	 _zj = svdup_lane(_jp, 14);

      _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

	 _mj = svdup_lane(_jp, 15);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);

         _dx = svsub_x(pg, _xj, _xi);
	 _dy = svsub_x(pg, _yj, _yi);
	 _dz = svsub_x(pg, _zj, _zi);
      
	 _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
	 _rsq = svmad_x(pg, _dy, _dy, _rsq);
	 _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
	 _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
	 _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
	 _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
	 _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
	 _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

	 _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
	 _mrinv3 = svmul_x(pg, _mj, _mrinv3);
	 _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

	 _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
	 _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
	 _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);
    }

    for(int j=nj16;j<nj;j++) {
      _mj = svld1(pg_j, (float32_t *)(jp+j));
      _xj = svdup_lane(_mj, 0);
      _yj = svdup_lane(_mj, 1);
      _zj = svdup_lane(_mj, 2);
      _mj = svdup_lane(_mj, 3);

      _dx = svsub_x(pg, _xj, _xi);
      _dy = svsub_x(pg, _yj, _yi);
      _dz = svsub_x(pg, _zj, _zi);

      _rsq = svmad_x(pg, _dx, _dx, FLT_MIN);
      _rsq = svmad_x(pg, _dy, _dy, _rsq);
      _rsq = svmad_x(pg, _dz, _dz, _rsq);

#ifdef __APPROX_RCP_RSQRT__
      _rad = svmul_x(pg, svrsqrte(_rsq), _rsq);
#else
      _rad = svmul_x(pg, rsqrt_acc(pg, _rsq), _rsq);
#endif
      _gfact = gfactor_acc_sve(pg, _rad, sft_pm_inv);
#ifdef __APPROX_RCP_RSQRT__
      _mrinv3 = svrsqrte(svadd_x(pg, _rsq, _eps2)); // 1.0/sqrt(r^2 + eps^2)
#else
      _mrinv3 = rsqrt_acc(pg, svadd_x(pg, _rsq, _eps2));
#endif

      _mrinv3 = svmul_x(pg, svmul_x(pg, _mrinv3, _mrinv3), _mrinv3);
      _mrinv3 = svmul_x(pg, _mj, _mrinv3);
      _mrinv3 = svmul_x(pg, _gfact, _mrinv3);

      _xacc = svmad_x(pg, _mrinv3, _dx, _xacc);
      _yacc = svmad_x(pg, _mrinv3, _dy, _yacc);
      _zacc = svmad_x(pg, _mrinv3, _dz, _zacc);
    }

    svst1(pg, result->xacc+i, _xacc);
    svst1(pg, result->yacc+i, _yacc);
    svst1(pg, result->zacc+i, _zacc);

    i += svcntw();
    pg = svwhilelt_b32(i, ni);
  }while(svptest_any(svptrue_b32(), pg));

}

void grav_S2_wo_sve(struct iptcl *ip, struct fodata *result, int ni,
		    struct jptcl *jp, int nj, float sft_pm)
{
  
  for(int i=0;i<ni;i++) {
    result->xacc[i] = 0.0;
    result->yacc[i] = 0.0;
    result->zacc[i] = 0.0;
  }


  for(int i=0;i<ni;i++) {
    for(int j=0;j<nj;j++) {
      float dx = jp[j].xpos-ip->xpos[i];
      float dy = jp[j].ypos-ip->ypos[i];
      float dz = jp[j].zpos-ip->zpos[i];

      float rsq = dx*dx + dy*dy + dz*dz;
      float rad = sqrt(rsq);
      float rinv = 1.0/sqrt(rsq+SQR(ip->eps[i]));
      float mrinv3 = jp[j].mass*CUBE(rinv);

      float gfact = gfactor_acc(rad, sft_pm);

      result->xacc[i] += mrinv3*dx*gfact;
      result->yacc[i] += mrinv3*dy*gfact;
      result->zacc[i] += mrinv3*dz*gfact;

    }
  }

}


void set_iptcl(struct iptcl *ip)
{
  for(int i=0;i<NIMAX;i++) {
    ip->xpos[i] = genrand_res53();
    ip->ypos[i] = genrand_res53();
    ip->zpos[i] = genrand_res53();
    ip->eps[i]  = 0.05;
  }
}

void set_jptcl(struct jptcl *jp, int nj)
{
  for(int j=0;j<nj;j++) {
    jp[j].xpos = genrand_res53();
    jp[j].ypos = genrand_res53();
    jp[j].zpos = genrand_res53();
    jp[j].mass = 1.0/nj;
  }
}

double timing(struct timespec from, struct timespec to)
{
  return ((to.tv_sec+(double)to.tv_nsec*1.e-9) - (from.tv_sec+(double)from.tv_nsec*1.e-9));
}

int main(int argc, char **argv)
{
  struct iptcl ip;
  struct jptcl *jp;
  struct fodata result_wo_sve;
  struct fodata result_with_sve;
  struct fodata result_with_sve_unroll16;

  struct timespec to, from;

  init_genrand(10);

  jp = (struct jptcl *)malloc(sizeof(struct jptcl)*NJMAX);

  //  int nj = 2048;
  int ni = 100;
  int nj = 2000;
  float sft_pm = 3.0/32.0;

  assert(ni <= NIMAX);
  assert(nj <= NJMAX);

  set_iptcl(&ip);
  set_jptcl(jp, nj);

  //======================================================
#ifdef __PROFILING__  
  fapp_start("nosve", 1, 0);
#endif
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &from);  
  grav_S2_wo_sve(&ip, &result_wo_sve, ni, jp, nj, sft_pm);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &to);
#ifdef __PROFILING__    
  fapp_stop("nosve", 1, 0);
#endif

  float cputime_wo_sve = timing(from, to);
  printf("WO_SVE:%14.6e [interactions/sec] \n",(double)(ni*nj)/cputime_wo_sve);
  //======================================================



  //======================================================
#ifdef __PROFILING__    
  fapp_start("sve", 1, 0);
#endif
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &from);    
  grav_S2_with_sve(&ip, &result_with_sve, ni, jp, nj, sft_pm);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &to);  
#ifdef __PROFILING__
  fapp_stop("sve", 1, 0);
#endif

  float cputime_with_sve = timing(from, to);
  printf("WITH_SVE:%14.6e [interactions/sec] \n",(double)(ni*nj)/cputime_with_sve);
  //======================================================


  //======================================================  
#ifdef __PROFILING__    
  fapp_start("sve_unroll16", 1, 0);
#endif
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &from);  
  grav_S2_with_sve_unroll16(&ip, &result_with_sve_unroll16, ni, jp, nj, sft_pm);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &to);
#ifdef __PROFILING__
  fapp_stop("sve_unroll16", 1, 0);
#endif

  float cputime_with_sve_unroll16 = timing(from, to);
  printf("WITH_SVE_UNROLL16:%14.6e [interactions/sec] \n",(double)(ni*nj)/cputime_with_sve_unroll16);
  //======================================================


#ifdef __CHECK_FORCE__
  for(int i=0;i<ni;i++) {
    printf("%d %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n",
	   i,
	   result_wo_sve.xacc[i], result_with_sve.xacc[i], result_with_sve_unroll16.xacc[i],
	   result_wo_sve.yacc[i], result_with_sve.yacc[i], result_with_sve_unroll16.yacc[i],
	   result_wo_sve.zacc[i], result_with_sve.zacc[i], result_with_sve_unroll16.zacc[i]);
  }
#endif
  
}
