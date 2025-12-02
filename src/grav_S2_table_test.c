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

// For the table look-up version
#define EXP_BIT (4)
#define FRC_BIT (5)

#define TICK (1<<(23-FRC_BIT))
#define TBL_SIZE (1<<(EXP_BIT+FRC_BIT))

// PM softening
#define SFT_FOR_PM (3.0/32.0)
#define SFT_FOR_PP (0.01/32.0)

float Force_table[TBL_SIZE][2];
float xmscale[16];

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

static inline REAL reff(double rad, double eps_pm)
{
  return 1.0/CUBE(rad)*gfactor_acc(rad, eps_pm);
}

static float R2scale;
static float Xscale;
static float Ascale;
static float R2cut_xscale2;
union pack32{
  float f;
  unsigned int u;
};

void pg5_gen_force_table(double (*force_func)(double rad), double rcut)
{
  union pack32 m32;
  const unsigned tick = (1<<(23-FRC_BIT));

  const float r2max = SQR(rcut);
  const float fmax = (1<<(1<<EXP_BIT))*(2.0-1.0/(1<<FRC_BIT));
  const float r2scale = (fmax-2.0f)/r2max;
  // so that 0 <= r2/r2max <= 2^17 - 3
  // or 2 <= s < 2^17, where s = r2/r2max + 2.0
  // in binary, 0_100000000_00000000... <= s <= 0_100001111_11110000.

  R2scale = r2scale;
  Xscale = sqrtf(r2scale);

  Ascale = 1.0f/Xscale;

  R2cut_xscale2 = (1<<(1<<EXP_BIT))*(2.0-1.0/(1<<FRC_BIT));

  xmscale[0]  = Xscale;
  xmscale[1]  = Xscale;
  xmscale[2]  = Xscale;
  xmscale[3]  = 1.0f;
  xmscale[4]  = Xscale;
  xmscale[5]  = Xscale;
  xmscale[6]  = Xscale;
  xmscale[7]  = 1.0f;
  xmscale[8]  = Xscale;
  xmscale[9]  = Xscale;
  xmscale[10] = Xscale;
  xmscale[11] = 1.0f;
  xmscale[12] = Xscale;
  xmscale[13] = Xscale;
  xmscale[14] = Xscale;
  xmscale[15] = 1.0f;

  int i;
  // table value and...
  for(i=0, m32.f = 2.0f;
      i<TBL_SIZE; 
      i++, m32.u += tick) {
    float f = m32.f;
    float r2 = (f - 2.0) / r2scale;
    float r = sqrtf(r2);
    Force_table[i][0] = force_func(r);
  }

  // ...slope
  for(i=0, m32.f = 2.0f; 
      i<TBL_SIZE-1; 
      i++) {
    float x0 = m32.f;
    m32.u += tick;
    float x1 = m32.f;
    float y0 = Force_table[i][0];
    float y1 = (i==TBL_SIZE-1) ? 0.0 : Force_table[i+1][0];
    Force_table[i][1] = (y1-y0)/(x1-x0);
  }
  Force_table[TBL_SIZE-1][1] = 0.0f;
}

double s2_force(double rad)
{
  return (reff(rad, SFT_FOR_PM)-reff(rad, SFT_FOR_PP));
}

static float refer_table(float r){
  union pack32 m32;
  float x = r*r*R2scale + 2.0f;
  m32.f = x;
  int idx = (m32.u >> (23-FRC_BIT)) & (TBL_SIZE-1);
  assert(idx < TBL_SIZE);
  m32.u >>= (23-FRC_BIT);
  m32.u <<= (23-FRC_BIT);
  float dx = (x - m32.f);
  return Force_table[idx][0] + dx*Force_table[idx][1];
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
//#define DEBUG
#ifdef DEBUG
        jp[j].xpos = genrand_res53() * Xscale;
        jp[j].ypos = genrand_res53() * Xscale;
        jp[j].zpos = genrand_res53() * Xscale;
#else
        jp[j].xpos = genrand_res53();
        jp[j].ypos = genrand_res53();
        jp[j].zpos = genrand_res53();
#endif
        jp[j].mass = 1.0/nj;
    }
}

void set_jptcl2(struct jptcl *jp0, struct jptcl *jp, int nj)
{
    for(int j=0;j<nj;j++) {
#ifdef DEBUG
        jp[j].xpos = jp0[j].xpos;
        jp[j].ypos = jp0[j].ypos;
        jp[j].zpos = jp0[j].zpos;
#else
        jp[j].xpos = jp0[j].xpos * Xscale;
        jp[j].ypos = jp0[j].ypos * Xscale;
        jp[j].zpos = jp0[j].zpos * Xscale;
#endif
        jp[j].mass = jp0[j].mass;
    }
}

void grav_S2_table_with_sve(struct iptcl *ip, struct fodata *result, int ni,
			    struct jptcl *jp, int nj, float sft_pm)
{
    //static unsigned int u32dbg[16], s32dbg[16];
    //static unsigned long long u64dbg[8], s64dbg[8];
    //static float f32dbg[16];
    
    //static unsigned int idx[16];
    
    uint32_t i = 0;
    svbool_t pg = svwhilelt_b32_u32(i, ni);
    svbool_t pg_j = svwhilelt_b32_u32(0, 4);
    
    svfloat32_t _xscale = svdup_f32(Xscale);
    svfloat32_t _r2cut_xscale2 = svdup_f32(R2cut_xscale2);
    svfloat32_t _ascale = svdup_f32(Ascale);
    //svfloat32_t _xmscale = svld1(svptrue_b32(), xmscale);
    
    int32_t *ptr = (int32_t *)Force_table-(1<<(31-(23-FRC_BIT)));
    
    do {
        svfloat32_t _xacc = svdup_f32(0.0);
        svfloat32_t _yacc = svdup_f32(0.0);
        svfloat32_t _zacc = svdup_f32(0.0);
        
        svfloat32_t _xi = svld1(pg, ip->xpos+i);
        _xi = svmul_x(pg, _xi, _xscale);
        svfloat32_t _yi = svld1(pg, ip->ypos+i);
        _yi = svmul_x(pg, _yi, _xscale);
        svfloat32_t _zi = svld1(pg, ip->zpos+i);
        _zi = svmul_x(pg, _zi, _xscale);

	
	svfloat32_t _xmj = svld1(pg_j, (float32_t *)jp);

	svfloat32_t _xj = svdup_lane(_xmj, 0);
	svfloat32_t _yj = svdup_lane(_xmj, 1);
	svfloat32_t _zj = svdup_lane(_xmj, 2);
	svfloat32_t _mj = svdup_lane(_xmj, 3);  

        for(int j=0;j<nj;j++) {
			    
            svfloat32_t _dx = svsub_z(pg, _xj, _xi);
            svfloat32_t _dy = svsub_z(pg, _yj, _yi);
            svfloat32_t _dz = svsub_z(pg, _zj, _zi);

            svfloat32_t _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            svuint32_t _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            svuint32_t _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);

	       _xmj = svld1(pg_j, (float32_t *)(jp+j+1));	    
            
            svfloat32_t _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            svfloat32_t _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));

	       _xj = svdup_lane(_xmj, 0);

            
            svfloat32_t _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));

	       _yj = svdup_lane(_xmj, 1);
	       
            _ff = svmad_x(pg, _df, _dr2, _ff);

	       _zj = svdup_lane(_xmj, 2);
	       
            _ff = svmul_x(pg, _ff, _mj);


	       _mj = svdup_lane(_xmj, 3);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
        }
        
        svst1(pg, result->xacc+i, svmul_x(pg, _xacc, _ascale));
        svst1(pg, result->yacc+i, svmul_x(pg, _yacc, _ascale));
        svst1(pg, result->zacc+i, svmul_x(pg, _zacc, _ascale));
        
        i+=svcntw();
        pg = svwhilelt_b32_u32(i,ni);
    } while(svptest_any(svptrue_b32(), pg));
}

void printVector(svbool_t pg, svfloat32_t _vec) {
    static float f32dbg[16];

    svst1(pg, f32dbg, _vec);
    for(int iii = 0; iii < 16; iii++) {
        printf(" %+e", f32dbg[iii]);
    }
    printf("\n");
}

void grav_S2_table_with_sve_4x4(struct iptcl *ip, struct fodata *result, int ni,
			    struct jptcl *jp, int nj, float sft_pm)
{
    const int32_t nip = 4;
    const int32_t njp = 4;
    static const uint32_t iperm[16] = {0, 0, 0, 0,
                                       1, 1, 1, 1,
                                       2, 2, 2, 2,
                                       3, 3, 3, 3};
    static const uint32_t jxperm[16] = {0, 4, 8, 12,
                                        0, 4, 8, 12,
                                        0, 4, 8, 12,
                                        0, 4, 8, 12};
    static const uint32_t jyperm[16] = {1, 5, 9, 13,
                                        1, 5, 9, 13,
                                        1, 5, 9, 13,
                                        1, 5, 9, 13};
    static const uint32_t jzperm[16] = {2, 6, 10, 14,
                                        2, 6, 10, 14,
                                        2, 6, 10, 14,
                                        2, 6, 10, 14};
    static const uint32_t jmperm[16] = {3, 7, 11, 15,
                                        3, 7, 11, 15,
                                        3, 7, 11, 15,
                                        3, 7, 11, 15};
    const svuint32_t  _iperm  = svld1(svptrue_b32(), iperm);
    const svuint32_t  _jxperm = svld1(svptrue_b32(), jxperm);
    const svuint32_t  _jyperm = svld1(svptrue_b32(), jyperm);
    const svuint32_t  _jzperm = svld1(svptrue_b32(), jzperm);
    const svuint32_t  _jmperm = svld1(svptrue_b32(), jmperm);

    static const uint32_t ops[16] = {0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0};
    static const uint32_t op0[16] = {0, 0, 0, 0, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1};
    static const uint32_t op1[16] = {1, 1, 1, 1, 0, 0, 0, 0,
                                     1, 1, 1, 1, 1, 1, 1, 1};
    static const uint32_t op2[16] = {1, 1, 1, 1, 1, 1, 1, 1,
                                     0, 0, 0, 0, 1, 1, 1, 1};
    static const uint32_t op3[16] = {1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 0, 0, 0, 0};
    const svuint32_t _ops = svld1(svptrue_b32(), ops);
    const svuint32_t _op0 = svld1(svptrue_b32(), op0);
    const svuint32_t _op1 = svld1(svptrue_b32(), op1);
    const svuint32_t _op2 = svld1(svptrue_b32(), op2);
    const svuint32_t _op3 = svld1(svptrue_b32(), op3);
    const svbool_t _rdc0  = svcmpeq(svptrue_b32(), _ops, _op0);
    const svbool_t _rdc1  = svcmpeq(svptrue_b32(), _ops, _op1);
    const svbool_t _rdc2  = svcmpeq(svptrue_b32(), _ops, _op2);
    const svbool_t _rdc3  = svcmpeq(svptrue_b32(), _ops, _op3);

    int32_t i = 0;
    svbool_t pg = svwhilelt_b32_u32(i, ni);
    svbool_t pg_i = svwhilelt_b32_u32(i*nip, ni*nip);
    
    svfloat32_t _xscale = svdup_f32(Xscale);
    svfloat32_t _r2cut_xscale2 = svdup_f32(R2cut_xscale2);
    svfloat32_t _ascale = svdup_f32(Ascale);

    int32_t *ptr = (int32_t *)Force_table-(1<<(31-(23-FRC_BIT)));

    do {
        int32_t j = 0;
        svbool_t pg_j = svwhilelt_b32_u32(j*njp, nj*njp);

        svfloat32_t _xacc = svdup_f32(0.0);
        svfloat32_t _yacc = svdup_f32(0.0);
        svfloat32_t _zacc = svdup_f32(0.0);

        svfloat32_t _xi = svtbl(svld1(pg, ip->xpos+i), _iperm);
        _xi = svmul_x(pg_i, _xi, _xscale);
        svfloat32_t _yi = svtbl(svld1(pg, ip->ypos+i), _iperm);
        _yi = svmul_x(pg_i, _yi, _xscale);
        svfloat32_t _zi = svtbl(svld1(pg, ip->zpos+i), _iperm);
        _zi = svmul_x(pg_i, _zi, _xscale);
        
        float32_t * jptr = (float32_t *)jp;

            svfloat32_t _xmj = svld1(pg_j, jptr+4*j);

            svfloat32_t _xj = svtbl(_xmj, _jxperm);
            svfloat32_t _yj = svtbl(_xmj, _jyperm);
            svfloat32_t _zj = svtbl(_xmj, _jzperm);
            svfloat32_t _mj = svtbl(_xmj, _jmperm);	

        do {
            
            svfloat32_t _dx = svsub_z(pg_i, _xj, _xi);
            svfloat32_t _dy = svsub_z(pg_i, _yj, _yi);
            svfloat32_t _dz = svsub_z(pg_i, _zj, _zi);

            svfloat32_t _rsq = svmad_z(pg_i, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg_i, _dy, _dy, _rsq);
            _rsq = svmad_z(pg_i, _dz, _dz, _rsq);
            _rsq = svmin_z(pg_i, _rsq, _r2cut_xscale2);

            svuint32_t _rsq_sr = svlsr_z(pg_i, svreinterpret_u32(_rsq), 23-FRC_BIT);
            svuint32_t _rsq_sl = svlsl_z(pg_i, _rsq_sr,                 23-FRC_BIT);

	    svfloat32_t _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg_i, ptr,   svlsl_z(pg_i, _rsq_sr, 1)));
	    svfloat32_t _df = svreinterpret_f32(svld1_gather_u32index_s32(pg_i, ptr+1, svlsl_z(pg_i, _rsq_sr, 1)));

	       _xmj = svld1(pg_j, jptr+4*j);	    
            
            svfloat32_t _dr2 = svsub_x(pg_i, _rsq, svreinterpret_f32(_rsq_sl));	    
            _ff = svmad_x(pg_i, _df, _dr2, _ff);
            _ff = svmul_x(pg_i, _ff, _mj);

	       _xj = svtbl(_xmj, _jxperm);
	       _yj = svtbl(_xmj, _jyperm);
	       _zj = svtbl(_xmj, _jzperm);
	       _mj = svtbl(_xmj, _jmperm);

            _xacc = svmad_x(pg_i, _ff, _dx, _xacc);
            _yacc = svmad_x(pg_i, _ff, _dy, _yacc);
            _zacc = svmad_x(pg_i, _ff, _dz, _zacc);

            j   += njp;
            pg_j = svwhilelt_b32_u32(j*njp, nj*njp);
        } while(j < nj);

        svfloat32_t axbuf = svmul_x(pg_i, _xacc, _ascale);
        (result->xacc+i)[0] = svaddv(_rdc0, axbuf);
        (result->xacc+i)[1] = svaddv(_rdc1, axbuf);
        (result->xacc+i)[2] = svaddv(_rdc2, axbuf);
        (result->xacc+i)[3] = svaddv(_rdc3, axbuf);

        svfloat32_t aybuf = svmul_x(pg_i, _yacc, _ascale);
        (result->yacc+i)[0] = svaddv(_rdc0, aybuf);
        (result->yacc+i)[1] = svaddv(_rdc1, aybuf);
        (result->yacc+i)[2] = svaddv(_rdc2, aybuf);
        (result->yacc+i)[3] = svaddv(_rdc3, aybuf);

        svfloat32_t azbuf = svmul_x(pg_i, _zacc, _ascale);
        (result->zacc+i)[0] = svaddv(_rdc0, azbuf);
        (result->zacc+i)[1] = svaddv(_rdc1, azbuf);
        (result->zacc+i)[2] = svaddv(_rdc2, azbuf);
        (result->zacc+i)[3] = svaddv(_rdc3, azbuf);

        i+=nip;
        pg = svwhilelt_b32_u32(i,ni);
        pg_i = svwhilelt_b32_u32(i*nip, ni*nip);
    } while(i < ni);
}

void grav_S2_table_with_sve_unroll16(struct iptcl *ip, struct fodata *result, int ni,
				      struct jptcl *jp, int nj, float sft_pm)
{
    uint32_t i = 0;
    svbool_t pg = svwhilelt_b32_u32(i, ni);
    svbool_t pg_j = svptrue_b32();
    
    static unsigned int idx[16];
    
    svfloat32_t _xscale = svdup_f32(Xscale);
    svfloat32_t _r2cut_xscale2 = svdup_f32(R2cut_xscale2);
    svfloat32_t _ascale = svdup_f32(Ascale);
    //svfloat32_t _xmscale = svld1(svptrue_b32(), xmscale);
    
    int32_t *ptr = (int32_t *)Force_table-(1<<(31-(23-FRC_BIT)));
    
    int nj16, mod_nj16;
    mod_nj16 = nj%16;
    nj16 = nj - mod_nj16;
    
    do {
        svfloat32_t _xacc = svdup_f32(0.0);
        svfloat32_t _yacc = svdup_f32(0.0);
        svfloat32_t _zacc = svdup_f32(0.0);
        
        svfloat32_t _xi = svld1(pg, ip->xpos+i);
        _xi = svmul_x(pg, _xi, _xscale);
        svfloat32_t _yi = svld1(pg, ip->ypos+i);
        _yi = svmul_x(pg, _yi, _xscale);
        svfloat32_t _zi = svld1(pg, ip->zpos+i);
        _zi = svmul_x(pg, _zi, _xscale);
        
        
        for(int j=0;j<nj;j+=16) {
            svfloat32_t _rsq;
            svfloat32_t _ff, _df, _dr2;
            svfloat32_t _xj, _yj, _zj, _mj;
            svfloat32_t _dx, _dy, _dz;
            
            svuint32_t _rsq_sr, _rsq_sl;
            
            svfloat32_t _jp = svld1(pg_j, (float32_t *)(jp+j));
            //_jp = svmul_z(pg_j, _jp, _xmscale);
            
            // 0
            _xj = svdup_lane(_jp, 0);
            _yj = svdup_lane(_jp, 1);
            _zj = svdup_lane(_jp, 2);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _mj = svdup_lane(_jp, 3);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            // 1
            _xj = svdup_lane(_jp, 4);
            _yj = svdup_lane(_jp, 5);
            _zj = svdup_lane(_jp, 6);
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 7);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            // 2
            _xj = svdup_lane(_jp, 8);
            _yj = svdup_lane(_jp, 9);
            _zj = svdup_lane(_jp, 10);
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 11);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            // 3
            _xj = svdup_lane(_jp, 12);
            _yj = svdup_lane(_jp, 13);
            _zj = svdup_lane(_jp, 14);      
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 15);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            // 4
            _jp = svld1(pg_j, (float32_t *)(jp+j+4));
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            //_jp = svmul_z(pg_j, _jp, _xmscale);	
            _xj = svdup_lane(_jp, 0);
            _yj = svdup_lane(_jp, 1);
            _zj = svdup_lane(_jp, 2);
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 3);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            // 5
            _xj = svdup_lane(_jp, 4);
            _yj = svdup_lane(_jp, 5);
            _zj = svdup_lane(_jp, 6);
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 7);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            // 6
            _xj = svdup_lane(_jp, 8);
            _yj = svdup_lane(_jp, 9);
            _zj = svdup_lane(_jp, 10);
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 11);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            // 7
            _xj = svdup_lane(_jp, 12);
            _yj = svdup_lane(_jp, 13);
            _zj = svdup_lane(_jp, 14);      
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 15);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            // 8
            _jp = svld1(pg_j, (float32_t *)(jp+j+8));
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            //_jp = svmul_z(pg_j, _jp, _xmscale);	
            _xj = svdup_lane(_jp, 0);
            _yj = svdup_lane(_jp, 1);
            _zj = svdup_lane(_jp, 2);
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 3);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            // 9
            _xj = svdup_lane(_jp, 4);
            _yj = svdup_lane(_jp, 5);
            _zj = svdup_lane(_jp, 6);
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 7);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            // 10
            _xj = svdup_lane(_jp, 8);
            _yj = svdup_lane(_jp, 9);
            _zj = svdup_lane(_jp, 10);
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 11);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            // 11
            _xj = svdup_lane(_jp, 12);
            _yj = svdup_lane(_jp, 13);
            _zj = svdup_lane(_jp, 14);      
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 15);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);

            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            // 12
            _jp = svld1(pg_j, (float32_t *)(jp+j+12));
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            //_jp = svmul_z(pg_j, _jp, _xmscale);	
            _xj = svdup_lane(_jp, 0);
            _yj = svdup_lane(_jp, 1);
            _zj = svdup_lane(_jp, 2);
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 3);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            // 13
            _xj = svdup_lane(_jp, 4);
            _yj = svdup_lane(_jp, 5);
            _zj = svdup_lane(_jp, 6);
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 7);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            // 14
            _xj = svdup_lane(_jp, 8);
            _yj = svdup_lane(_jp, 9);
            _zj = svdup_lane(_jp, 10);
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 11);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            // 15
            _xj = svdup_lane(_jp, 12);
            _yj = svdup_lane(_jp, 13);
            _zj = svdup_lane(_jp, 14);      
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _mj = svdup_lane(_jp, 15);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
            
            _dx = svsub_z(pg, _xj, _xi);
            _dy = svsub_z(pg, _yj, _yi);
            _dz = svsub_z(pg, _zj, _zi);
            
            _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);
            
            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
        }
        
        for(int j=nj16;j<nj;j++) {
            svfloat32_t _mj = svld1(pg_j, (float32_t *)(jp+j));
            
            //_mj = svmul_x(pg_j, _mj, _xmscale);
            svfloat32_t _xj = svdup_lane(_mj, 0);
            svfloat32_t _yj = svdup_lane(_mj, 1);
            svfloat32_t _zj = svdup_lane(_mj, 2);
            _mj = svdup_lane(_mj, 3);
            
            svfloat32_t _dx = svsub_z(pg, _xj, _xi);
            svfloat32_t _dy = svsub_z(pg, _yj, _yi);
            svfloat32_t _dz = svsub_z(pg, _zj, _zi);
            
            svfloat32_t _rsq = svmad_z(pg, _dx, _dx, 2.0f);
            _rsq = svmad_z(pg, _dy, _dy, _rsq);
            _rsq = svmad_z(pg, _dz, _dz, _rsq);
            _rsq = svmin_z(pg, _rsq, _r2cut_xscale2);
            
            svuint32_t _rsq_sr = svlsr_z(pg, svreinterpret_u32(_rsq), 23-FRC_BIT);
            svst1(pg, idx, _rsq_sr);
            svuint32_t _rsq_sl = svlsl_z(pg, _rsq_sr, 23-FRC_BIT);
            
            svfloat32_t _ff = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr, svlsl_z(pg, _rsq_sr, 1)));
            svfloat32_t _df = svreinterpret_f32(svld1_gather_u32index_s32(pg, ptr+1, svlsl_z(pg, _rsq_sr, 1)));
            
            svfloat32_t _dr2 = svsub_x(pg, _rsq, svreinterpret_f32(_rsq_sl));
            _ff = svmad_x(pg, _df, _dr2, _ff);
            _ff = svmul_x(pg, _ff, _mj);

            _xacc = svmad_x(pg, _ff, _dx, _xacc);
            _yacc = svmad_x(pg, _ff, _dy, _yacc);
            _zacc = svmad_x(pg, _ff, _dz, _zacc);
        }
        
        svst1(pg, result->xacc+i, svmul_x(pg, _xacc, _ascale));
        svst1(pg, result->yacc+i, svmul_x(pg, _yacc, _ascale));
        svst1(pg, result->zacc+i, svmul_x(pg, _zacc, _ascale));
        
        i+=svcntw();
    pg = svwhilelt_b32_u32(i,ni);
    } while(svptest_any(svptrue_b32(), pg));
    
}

void grav_S2_table_wo_sve(struct iptcl *ip, struct fodata *result, int ni,
                          struct jptcl *jp, int nj, float sft_pm)
{
    
    for(int i=0;i<ni;i++) {
        result->xacc[i] = 0.0;
        result->yacc[i] = 0.0;
        result->zacc[i] = 0.0;
    }

    float xsinv = 1. / Xscale;
    
    for(int i=0;i<ni;i++) {
        for(int j=0;j<nj;j++) {
            float dx = jp[j].xpos*xsinv-ip->xpos[i];
            float dy = jp[j].ypos*xsinv-ip->ypos[i];
            float dz = jp[j].zpos*xsinv-ip->zpos[i];
            
            float rsq = dx*dx + dy*dy + dz*dz;
            float rad = sqrt(rsq);
            rad = fmin(rad, sft_pm);
            
            float mref = refer_table(rad)*jp[j].mass;
            
            result->xacc[i] += mref*dx;
            result->yacc[i] += mref*dy;
            result->zacc[i] += mref*dz;
        }
    }
    
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
            float s2 = s2_force((double)rad);
            
            float mref = s2*jp[j].mass;
            
            result->xacc[i] += mref*dx;
            result->yacc[i] += mref*dy;
            result->zacc[i] += mref*dz;
        }
    }
    
}

double timing(struct timespec from, struct timespec to)
{
  return ((to.tv_sec+(double)to.tv_nsec*1.e-9) - (from.tv_sec+(double)from.tv_nsec*1.e-9));
}

int main(int argc, char **argv)
{
  struct iptcl ip;
  struct jptcl *jp0, *jp;
  struct fodata result_wo_sve;
  struct fodata result_with_sve;
  struct fodata result_with_sve_unroll16;
  struct fodata result_with_sve_4x4;

  struct timespec to, from;

  init_genrand(10);

  jp0 = (struct jptcl *)malloc(sizeof(struct jptcl)*NJMAX);
  jp  = (struct jptcl *)malloc(sizeof(struct jptcl)*NJMAX);

  //  int nj = 2048;
  int ni = 100; //1000;
  int nj = 20000;
  //float sft_pm = SFT_FOR_PM;

  assert(ni <= NIMAX);
  assert(nj <= NJMAX);

  pg5_gen_force_table(s2_force, SFT_FOR_PM);

  set_iptcl(&ip);
  set_jptcl(jp0, nj);

  //======================================================  
#ifdef __PROFILING__  
  fapp_start("nosve", 1, 0);
#endif  
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &from);
  set_jptcl2(jp0, jp, nj);
  grav_S2_table_wo_sve(&ip, &result_wo_sve, ni, jp, nj, SFT_FOR_PM);
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
  set_jptcl2(jp0, jp, nj);
  grav_S2_table_with_sve(&ip, &result_with_sve, ni, jp, nj, SFT_FOR_PM);
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
  set_jptcl2(jp0, jp, nj);
  grav_S2_table_with_sve_unroll16(&ip, &result_with_sve_unroll16, ni, jp, nj, SFT_FOR_PM);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &to);
#ifdef __PROFILING__    
  fapp_stop("sve_unroll16", 1, 0);
#endif

  float cputime_with_sve_unroll16 = timing(from, to);
  printf("WITH_SVE_UNROLL16:%14.6e [interactions/sec] \n",(double)(ni*nj)/cputime_with_sve_unroll16);
  //======================================================  

  
  //======================================================
#ifdef __PROFILING__    
  fapp_start("sve_4x4", 1, 0);
#endif  
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &from);    
  set_jptcl2(jp0, jp, nj);
  grav_S2_table_with_sve_4x4(&ip, &result_with_sve_4x4, ni, jp, nj, SFT_FOR_PM);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &to);
#ifdef __PROFILING__    
  fapp_stop("sve_4x4", 1, 0);
#endif

  float cputime_with_sve_4x4 = timing(from, to);
  printf("WITH_SVE_4X4:%14.6e [interactions/sec] \n",(double)(ni*nj)/cputime_with_sve_4x4);
  //======================================================  

  
#ifdef __CHECK_FORCE__
  for(int i=0;i<1;i++) {
    printf("%d %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e %14.6e\n",
	   i,
	   result_wo_sve.xacc[i], result_with_sve.xacc[i], result_with_sve_unroll16.xacc[i], result_with_sve_4x4.xacc[i],
	   result_wo_sve.yacc[i], result_with_sve.yacc[i], result_with_sve_unroll16.yacc[i], result_with_sve_4x4.yacc[i],
	   result_wo_sve.zacc[i], result_with_sve.zacc[i], result_with_sve_unroll16.zacc[i], result_with_sve_4x4.zacc[i]);
  }
#endif

#ifdef __CHECK_TABLE__
#define NRAD_SAMPLE (1024)
  float dr = (SFT_FOR_PM-SFT_FOR_PP)/NRAD_SAMPLE;
  for(int ir=0;ir<NRAD_SAMPLE;ir++) {
    float rad = SFT_FOR_PP + dr*(float)ir;

    printf("%14.6e %14.6e %14.6e\n", rad, refer_table(rad), s2_force(rad));
  }
#endif
}
