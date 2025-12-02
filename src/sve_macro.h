#define DBG_PRINT_FLOAT(PREDICATE, X, Y)	                        \
  svst1(PREDICATE, f32dbg, X);                                          \
  printf("%s: ", Y);                                                    \
  for(int ii=0;ii<svcntw();ii++) printf("%14.6e  ", f32dbg[ii]);     	\
  printf("\n");

#define DBG_PRINT_UINT32(PREDICATE, X, Y)		                \
  svst1(PREDICATE, u32dbg, X);                                          \
  printf("%s: ", Y);                                                    \
  for(int ii=0;ii<svcntw();ii++) printf("%d  ", u32dbg[ii]);	        \
  printf("\n");

#define DBG_PRINT_SINT32(PREDICATE, X, Y)		                \
  svst1(PREDICATE, s32dbg, X);                                          \
  printf("%s: ", Y);                                                    \
  for(int ii=0;ii<svcntw();ii++) printf("%d  ", s32dbg[ii]);	        \
  printf("\n");

#define DBG_PRINT_UINT64(PREDICATE, X, Y)		                \
  svst1(PREDICATE, u64dbg, X);                                          \
  printf("%s: ", Y);                                                    \
  for(int ii=0;ii<svcntd();ii++) printf("%ld ", u64dbg[ii]);	        \
  printf("\n");

#define DBG_PRINT_SINT64(PREDICATE, X, Y)		                \
  svst1(PREDICATE, s64dbg, X);                                          \
  printf("%s: ", Y);                                                    \
  for(int ii=0;ii<svcntd();ii++) printf("%ld ", s64dbg[ii]);	        \
  printf("\n");
