CC=fccpx
#CFLAGS=-Nclang -Ofast
CFLAGS=-Kfast

#CFLAGS+=-D__PROFILING__
CFLAGS+=-D__CHECK_FORCE__

all: grav_S2_table_test grav_S2_test

GRAV_S2_TABLE_TEST_OBJ = grav_S2_table_test.o
GRAV_S2_TABLE_TEST_DEP := $(GRAV_S2_TABLE_TEST_OBJ)

GRAV_S2_TEST_OBJ = grav_S2_test.o
GRAV_S2_TEST_DEP := $(GRAV_S2_TEST_OBJ)

grav_S2_table_test: $(GRAV_S2_TABLE_TEST_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lm

grav_S2_test: $(GRAV_S2_TEST_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lm

clean:
	\rm -rf  *.o

distclean: clean
	\rm -rf grav_S2_table_test
	\rm -rf grav_S2_test
	\rm -rf *~



