#include "EuropeanOption.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {
bool ExpectClose(const std::string& label, double actual, double expected, double tolerance)
{
      if (std::abs(actual - expected) <= tolerance) {
            return true;
      }

      std::cerr << label << " failed. Expected " << expected << ", got " << actual << ".\n";
      return false;
}
}

int main()
{
      bool ok = true;

      EuropeanOption call("C");
      call.U = 100.0;
      call.K = 100.0;
      call.T = 1.0;
      call.r = 0.05;
      call.sig = 0.20;
      call.b = call.r;

      EuropeanOption put("P");
      put = call;
      put.SetOptionType("P");

      const double parityLeft = call.Price() - put.Price();
      const double parityRight = call.U * std::exp((call.b - call.r) * call.T)
                               - call.K * std::exp(-call.r * call.T);
      ok = ExpectClose("Put-call parity", parityLeft, parityRight, 1e-10) && ok;
      ok = ExpectClose("Gamma parity", call.getGamma(), put.getGamma(), 1e-12) && ok;

      EuropeanOption ivCall("C");
      ivCall.U = 100.0;
      ivCall.K = 100.0;
      ivCall.T = 1.0;
      ivCall.r = 0.05;
      ivCall.sig = 0.35;
      ivCall.b = ivCall.r;
      const double callMarketPrice = ivCall.Price();
      ivCall.sig = 0.10;
      const ImpliedVolatilityResult callIv = ivCall.SolveImpliedVolatility(callMarketPrice);
      ok = (callIv.success && ExpectClose("Call implied volatility", callIv.volatility, 0.35, 1e-7)) && ok;
      ivCall.sig = callIv.volatility;
      ok = ExpectClose("Call IV repricing", ivCall.Price(), callMarketPrice, 1e-6) && ok;

      EuropeanOption ivPut("P");
      ivPut = ivCall;
      ivPut.SetOptionType("P");
      ivPut.sig = 0.28;
      const double putMarketPrice = ivPut.Price();
      ivPut.sig = 0.12;
      const ImpliedVolatilityResult putIv = ivPut.SolveImpliedVolatility(putMarketPrice);
      ok = (putIv.success && ExpectClose("Put implied volatility", putIv.volatility, 0.28, 1e-7)) && ok;
      ivPut.sig = putIv.volatility;
      ok = ExpectClose("Put IV repricing", ivPut.Price(), putMarketPrice, 1e-6) && ok;

      const ImpliedVolatilityResult invalidIv =
          ivCall.SolveImpliedVolatility(ivCall.MaximumArbitragePrice() + 1.0);
      ok = ExpectClose("Invalid IV rejection", invalidIv.success ? 1.0 : 0.0, 0.0, 0.0) && ok;

      EuropeanOption europeanCall("C");
      europeanCall.U = 100.0;
      europeanCall.K = 100.0;
      europeanCall.T = 1.0;
      europeanCall.r = 0.05;
      europeanCall.sig = 0.20;
      europeanCall.b = europeanCall.r;

      EuropeanOption americanCall = europeanCall;
      americanCall.SetExerciseStyle("A");
      americanCall.binomialSteps = 1000;
      ok = ExpectClose("American non-dividend call matches European",
                       americanCall.Price(),
                       europeanCall.Price(),
                       2e-2) && ok;

      EuropeanOption europeanPut("P");
      europeanPut.U = 40.0;
      europeanPut.K = 50.0;
      europeanPut.T = 1.0;
      europeanPut.r = 0.10;
      europeanPut.sig = 0.25;
      europeanPut.b = europeanPut.r;

      EuropeanOption americanPut = europeanPut;
      americanPut.SetExerciseStyle("A");
      americanPut.binomialSteps = 1000;
      ok = ExpectClose("American put dominates European put",
                       americanPut.Price() >= europeanPut.Price() ? 1.0 : 0.0,
                       1.0,
                       0.0) && ok;

      EuropeanOption maturedCall("C");
      maturedCall.U = 120.0;
      maturedCall.K = 100.0;
      maturedCall.T = 0.0;
      maturedCall.sig = 0.25;
      maturedCall.r = 0.03;
      maturedCall.b = 0.03;
      ok = ExpectClose("Matured call intrinsic", maturedCall.Price(), 20.0, 1e-12) && ok;

      EuropeanOption maturedPut("P");
      maturedPut.U = 80.0;
      maturedPut.K = 100.0;
      maturedPut.T = 0.0;
      maturedPut.sig = 0.25;
      maturedPut.r = 0.03;
      maturedPut.b = 0.03;
      ok = ExpectClose("Matured put intrinsic", maturedPut.Price(), 20.0, 1e-12) && ok;

      EuropeanOption deterministicCall("C");
      deterministicCall.U = 100.0;
      deterministicCall.K = 95.0;
      deterministicCall.T = 0.5;
      deterministicCall.sig = 0.0;
      deterministicCall.r = 0.04;
      deterministicCall.b = 0.04;
      const double deterministicExpected = std::max(
          deterministicCall.U * std::exp((deterministicCall.b - deterministicCall.r) * deterministicCall.T) -
          deterministicCall.K * std::exp(-deterministicCall.r * deterministicCall.T),
          0.0);
      ok = ExpectClose("Zero-volatility price", deterministicCall.Price(), deterministicExpected, 1e-12) && ok;

      EuropeanOption toggled("C");
      toggled.toggle();
      ok = ExpectClose("Toggle to put", toggled.IsPut() ? 1.0 : 0.0, 1.0, 0.0) && ok;

      if (!ok) {
            return EXIT_FAILURE;
      }

      std::cout << "All regression checks passed.\n";
      return EXIT_SUCCESS;
}
