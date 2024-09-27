/*
Header file including necessary nvml headers.
*/

#pragma once

#include <ctime>
#include <cuda_runtime.h>
#include <nvml.h>
#include <thread>
#include <vector>

#include <memory>

class nvmlPower {
public:
  nvmlPower();
  ~nvmlPower();
 
 void startPolling();
 void stopPolling();

private:
  void powerPollingFunc();

  std::thread m_thread;
  bool pollstatus = false;
 
  nvmlReturn_t nvmlResult;
  nvmlEnableState_t pmmode;
  nvmlDevice_t nvmlDeviceID;
  std::vector<uint16_t> power;
};
