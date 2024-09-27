#include "tests/nvmlPower.h"
#include <chrono>
#include <iostream>
#include <cmath>
/*
These may be encompassed in a class if desired. Trivial CUDA programs written for the purpose of benchmarking might
prefer this approach.
*/

/*
Poll the GPU using nvml APIs.
*/
void nvmlPower::powerPollingFunc() {
  power.clear();
  unsigned int powerLevel = 0;
  uint16_t roundval=0;
  while (pollstatus) {
    nvmlResult = nvmlDeviceGetPowerUsage(nvmlDeviceID, &powerLevel);
    roundval=std::round(powerLevel / 1000.0);
    power.push_back(roundval);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }  
}
nvmlPower::nvmlPower() {
  // Initialize nvml.
  nvmlResult = nvmlInit();
  if (NVML_SUCCESS != nvmlResult) {
    std::cout << "NVML Init fail" << std::endl;
  }

  // This statement assumes that the first indexed GPU will be used.
  // If there are multiple GPUs that can be used by the system, this needs to be done with care.
  // Test thoroughly and ensure the correct device ID is being used.
  nvmlResult = nvmlDeviceGetHandleByIndex(0, &nvmlDeviceID);
  
  m_thread = std::thread();
  std::cout<<"NVML background thread started"<<std::endl;
}

nvmlPower::~nvmlPower() {
      std::cout<<"stopped esec"<<std::endl;
    stopPolling();  

  m_thread.join();
  nvmlResult = nvmlShutdown();
  if (NVML_SUCCESS != nvmlResult) {
    std::cout << "cannot shutdown nvml" << std::endl;
  }
}


void nvmlPower::startPolling() { 
  m_thread = std::thread(&nvmlPower::powerPollingFunc, this);
  pollstatus = true;
}

void nvmlPower::stopPolling() { 
  pollstatus = false; 
  
  m_thread.join();
  for (auto p:power)
    std::cout <<" "<< p;
}
