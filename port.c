#include "port.h"

#include <time.h>

int64_t get_nanotime(void)
   {
#ifdef _WIN32

   static LARGE_INTEGER ticksPerSec = {0};
   LARGE_INTEGER ticks;

   if (!ticksPerSec.QuadPart) 
      {
      QueryPerformanceFrequency(&ticksPerSec);
      if (!ticksPerSec.QuadPart) 
         return errno = ENOTSUP,-1;
      }

   QueryPerformanceCounter(&ticks);
   return ticks.QuadPart * 1000000000 / ticksPerSec.QuadPart;

#else

   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC,&t);
   return t.tv_sec * 1000000000 + t.tv_nsec;

#endif // _WIN32
   }

void set_thread_processor(int pnum)
   {
#ifdef _WIN32
   SetThreadAffinityMask(GetCurrentThread(),1LL << pnum);
#else
   cpu_set_t cpuset;
   CPU_ZERO(&cpuset);
   CPU_SET(pnum, &cpuset);
   pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
   }

int get_thread_processor()
   {
#ifdef _WIN32
   return GetCurrentProcessorNumber();
#else
   return sched_getcpu();
#endif
   }

THREAD_ID_TYPE start_thread(CThreadRoutine func,void *params,int pnum)
   {
   THREAD_ID_TYPE rv;
#ifdef _WIN32
   rv = CreateThread(NULL,0,func,params,CREATE_SUSPENDED,NULL);
   if (rv)
      {
      SetThreadAffinityMask(rv,1LL << pnum);
      ResumeThread(rv);
      }
   return rv;
#else
   if (!pthread_create(&rv,NULL,func,params))
      {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(pnum, &cpuset);
      pthread_setaffinity_np(rv, sizeof(cpuset), &cpuset);
      }
   return rv;   
#endif
   }

void wait_thread(THREAD_ID_TYPE thread)
   {
#ifdef _WIN32
   WaitForSingleObject(thread,INFINITE);
#else
   pthread_join(thread,NULL);
#endif
   }