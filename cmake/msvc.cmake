############################
# Definitions
############################
#/MP multiprocessor definition
add_definitions( -DUNICODE -D_UNICODE -DGLOG_NO_ABBREVIATED_SEVERITIES )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
 set(CMAKE_CXX_EXTENSIONS  OFF)

string (APPEND CMAKE_CXX_FLAGS  " /MP")
string (APPEND CMAKE_CXX_FLAGS  " /arch:AVX2")
string (APPEND CMAKE_CXX_FLAGS_DEBUG  " /Z7")
string (APPEND CMAKE_CXX_FLAGS  " /diagnostics:caret")

set (MSVC_DEPENDENCIES_DEBUG 
 #microsoft
  "${APIS_PATH}/Redist/MSVC/14.11.25325/debug_nonredist/x64/Microsoft.VC141.DebugCRT/msvcp140d.dll"
  "${APIS_PATH}/Redist/MSVC/14.11.25325/debug_nonredist/x64/Microsoft.VC141.DebugCRT/vccorlib140d.dll" 
    "${APIS_PATH}/Redist/MSVC/14.11.25325/debug_nonredist/x64/Microsoft.VC141.DebugCRT/vcruntime140d.dll" 
	 "${APIS_PATH}/Redist/MSVC/14.11.25325/debug_nonredist/x64/Microsoft.VC141.DebugCRT/concrt140d.dll" 
  )
 
set (MSVC_DEPENDENCIES_RELEASE
 
  "${APIS_PATH}/Redist/MSVC/14.11.25325/x64/Microsoft.VC141.CRT/msvcp140.dll"
  "${APIS_PATH}/Redist/MSVC/14.11.25325/x64/Microsoft.VC141.CRT/vccorlib140.dll" 
"${APIS_PATH}/Redist/MSVC/14.11.25325/x64/Microsoft.VC141.CRT/vcruntime140.dll" 
"${APIS_PATH}/Redist/MSVC/14.11.25325/x64/Microsoft.VC141.CRT/concrt140.dll" 
  )