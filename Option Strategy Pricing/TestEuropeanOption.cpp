#include "EuropeanOption.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr int kMinimumArgumentCount = 15;
constexpr int kArgumentCountWithMarketPrices = 17;

struct OptionInput {
      double spot;
      double strike;
      double volatility;
      double expiry;
      double dividend;
      double rate;
      std::string type;
      std::string style;
};

struct StrategyLeg {
      EuropeanOption option;
      double premium;
};

double absoluteDifference(double left, double right)
{
      return std::abs(left - right);
}

void PrintUsage(const char* programName)
{
      std::cout << "Usage:\n"
                << "  " << programName
                << " spot1 strike1 vol1 expiry1 dividend1 rate1 type1"
                << " spot2 strike2 vol2 expiry2 dividend2 rate2 type2"
                << " [style1 style2] [market_price1 market_price2]\n\n"
                << "Example:\n"
                << "  " << programName
                << " 100 95 0.25 0.5 0.01 0.04 C 100 105 0.25 0.5 0.01 0.04 C\n"
                << "  " << programName
                << " 100 100 0.20 1.0 0.00 0.05 C 100 100 0.20 1.0 0.00 0.05 P A A\n"
                << "  " << programName
                << " 100 100 0.20 1.0 0.00 0.05 C 100 100 0.20 1.0 0.00 0.05 P E E 10.4506 5.5735\n";
}

OptionInput ParseOptionInput(char* argv[], int startIndex)
{
      return {
          std::stod(argv[startIndex]),
          std::stod(argv[startIndex + 1]),
          std::stod(argv[startIndex + 2]),
          std::stod(argv[startIndex + 3]),
          std::stod(argv[startIndex + 4]),
          std::stod(argv[startIndex + 5]),
          argv[startIndex + 6],
          "E"
      };
}

EuropeanOption BuildOption(const OptionInput& input)
{
      EuropeanOption option;
      option.SetOptionType(input.type);
      option.SetExerciseStyle(input.style);
      option.U = input.spot;
      option.K = input.strike;
      option.sig = input.volatility;
      option.T = input.expiry;
      option.r = input.rate;
      option.b = input.rate - input.dividend;
      return option;
}

bool IsExerciseStyleToken(const std::string& token)
{
      return token == "E" || token == "e" || token == "A" || token == "a";
}

void PrintOptionSummary(const std::string& label,
                        const EuropeanOption& option,
                        bool hasMarketPrice,
                        double marketPrice,
                        const ImpliedVolatilityResult* iv)
{
      std::cout << label << '\n'
                << "  Type: " << option.optType << '\n'
                << "  Style: " << option.exerciseStyle << '\n'
                << "  Strike: " << option.K << '\n';
      if (hasMarketPrice) {
            std::cout << "  Model price: " << option.Price() << '\n';
      } else {
            std::cout << "  Price: " << option.Price() << '\n';
      }
      std::cout
                << "  Delta: " << option.Delta() << '\n'
                << "  Gamma: " << option.getGamma() << '\n'
                << "  Theta (daily): " << option.Theta() << '\n'
                << "  Vega: " << option.getVega() << '\n';
      if (hasMarketPrice) {
            std::cout << "  Market price: " << marketPrice << '\n';
            if (iv != nullptr && iv->success) {
                  std::cout << "  Implied volatility: " << iv->volatility
                            << " (" << iv->iterations << " iterations)\n";
            } else {
                  const std::string error = (iv != nullptr) ? iv->error : "not computed";
                  std::cout << "  Implied volatility: unavailable (" << error << ")\n";
            }
      }
      std::cout << '\n';
}

StrategyLeg BuildStrategyLeg(const EuropeanOption& option, double premium)
{
      return {option, premium};
}

void PrintStrategyBasis(bool usingMarketData)
{
      if (usingMarketData) {
            std::cout << "Strategy basis: market premiums with IV-implied Greeks\n\n";
      } else {
            std::cout << "Strategy basis: model premiums and Greeks from input volatility\n\n";
      }
}

void PrintNetGreeks(const EuropeanOption& longOption, const EuropeanOption& shortOption)
{
      std::cout << "Net Greeks:\n"
                << "  Delta: " << longOption.Delta() - shortOption.Delta() << '\n'
                << "  Gamma: " << longOption.getGamma() - shortOption.getGamma() << '\n'
                << "  Theta (daily): " << longOption.Theta() - shortOption.Theta() << '\n'
                << "  Vega: " << longOption.getVega() - shortOption.getVega() << '\n';
}

void PrintCombinedGreeks(const EuropeanOption& firstOption, const EuropeanOption& secondOption)
{
      std::cout << "Net Greeks:\n"
                << "  Delta: " << firstOption.Delta() + secondOption.Delta() << '\n'
                << "  Gamma: " << firstOption.getGamma() + secondOption.getGamma() << '\n'
                << "  Theta (daily): " << firstOption.Theta() + secondOption.Theta() << '\n'
                << "  Vega: " << firstOption.getVega() + secondOption.getVega() << '\n';
}

void PrintVerticalSpread(const std::string& name,
                         const StrategyLeg& longLeg,
                         const StrategyLeg& shortLeg,
                         double maxProfit,
                         double maxLoss,
                         double breakeven)
{
      std::cout << name << '\n'
                << "  Long strike: " << longLeg.option.K << '\n'
                << "  Short strike: " << shortLeg.option.K << '\n'
                << "  Net premium (debit positive / credit negative): "
                << longLeg.premium - shortLeg.premium << '\n';
      PrintNetGreeks(longLeg.option, shortLeg.option);
      std::cout << "  Max profit: " << maxProfit << '\n'
                << "  Max loss: " << maxLoss << '\n'
                << "  Breakeven: " << breakeven << "\n\n";
}

void PrintStraddleOrStrangle(const std::string& name,
                             const StrategyLeg& callLeg,
                             const StrategyLeg& putLeg)
{
      const double premium = callLeg.premium + putLeg.premium;
      std::cout << name << '\n';
      PrintCombinedGreeks(callLeg.option, putLeg.option);
      std::cout << "  Net premium: " << premium << '\n'
                << "  Max profit: unlimited\n"
                << "  Max loss: " << premium << '\n';

      if (absoluteDifference(callLeg.option.K, putLeg.option.K) < 1e-9) {
            std::cout << "  Breakevens: " << (callLeg.option.K - premium)
                      << " and " << (callLeg.option.K + premium) << "\n\n";
      } else {
            const double putStrike = putLeg.option.K;
            const double callStrike = callLeg.option.K;
            std::cout << "  Lower breakeven: " << (putStrike - premium) << '\n'
                      << "  Upper breakeven: " << (callStrike + premium) << "\n\n";
      }
}
}

int main(int argc, char* argv[])
{
      if (argc != 15 && argc != 17 && argc != 19) {
            PrintUsage(argv[0]);
            return 1;
      }

      OptionInput input1;
      OptionInput input2;
      bool hasStyles = false;
      bool hasMarketPrices = false;
      double marketPrice1 = 0.0;
      double marketPrice2 = 0.0;
      try {
            input1 = ParseOptionInput(argv, 1);
            input2 = ParseOptionInput(argv, 8);
            if (argc == 17) {
                  if (IsExerciseStyleToken(argv[15]) && IsExerciseStyleToken(argv[16])) {
                        hasStyles = true;
                        input1.style = argv[15];
                        input2.style = argv[16];
                  } else {
                        hasMarketPrices = true;
                        marketPrice1 = std::stod(argv[15]);
                        marketPrice2 = std::stod(argv[16]);
                  }
            } else if (argc == 19) {
                  hasStyles = true;
                  hasMarketPrices = true;
                  input1.style = argv[15];
                  input2.style = argv[16];
                  marketPrice1 = std::stod(argv[17]);
                  marketPrice2 = std::stod(argv[18]);
            }
      } catch (const std::exception&) {
            std::cout << "Could not parse the command-line inputs.\n\n";
            PrintUsage(argv[0]);
            return 1;
      }
      if (hasStyles && (!IsExerciseStyleToken(input1.style) || !IsExerciseStyleToken(input2.style))) {
            std::cout << "Style inputs must be E or A.\n\n";
            PrintUsage(argv[0]);
            return 1;
      }

      if (absoluteDifference(input1.spot, input2.spot) > 1e-9 ||
          absoluteDifference(input1.expiry, input2.expiry) > 1e-9) {
            std::cout << "Both options must share the same spot and expiry.\n\n";
            PrintUsage(argv[0]);
            return 1;
      }

      EuropeanOption option1 = BuildOption(input1);
      EuropeanOption option2 = BuildOption(input2);
      ImpliedVolatilityResult option1Iv{false, 0.0, 0, ""};
      ImpliedVolatilityResult option2Iv{false, 0.0, 0, ""};
      if (!option1.HasValidParameters()) {
            std::cout << "Option 1 is invalid: " << option1.ValidationError() << '\n';
            return 1;
      }
      if (!option2.HasValidParameters()) {
            std::cout << "Option 2 is invalid: " << option2.ValidationError() << '\n';
            return 1;
      }
      if (hasMarketPrices) {
            option1Iv = option1.SolveImpliedVolatility(marketPrice1);
            option2Iv = option2.SolveImpliedVolatility(marketPrice2);
      }

      StrategyLeg strategyLeg1 = BuildStrategyLeg(option1, option1.Price());
      StrategyLeg strategyLeg2 = BuildStrategyLeg(option2, option2.Price());
      bool usingMarketStrategyData = false;
      if (hasMarketPrices && option1Iv.success && option2Iv.success) {
            strategyLeg1.option.sig = option1Iv.volatility;
            strategyLeg2.option.sig = option2Iv.volatility;
            strategyLeg1.premium = marketPrice1;
            strategyLeg2.premium = marketPrice2;
            usingMarketStrategyData = true;
      }

      std::cout << std::fixed << std::setprecision(4);
      PrintOptionSummary("Option 1", option1, hasMarketPrices, marketPrice1, hasMarketPrices ? &option1Iv : nullptr);
      PrintOptionSummary("Option 2", option2, hasMarketPrices, marketPrice2, hasMarketPrices ? &option2Iv : nullptr);
      PrintStrategyBasis(usingMarketStrategyData);

      if (strategyLeg1.option.IsCall() && strategyLeg2.option.IsCall()) {
            const StrategyLeg& lowerStrikeCall = (strategyLeg1.option.K <= strategyLeg2.option.K) ? strategyLeg1 : strategyLeg2;
            const StrategyLeg& higherStrikeCall = (strategyLeg1.option.K <= strategyLeg2.option.K) ? strategyLeg2 : strategyLeg1;
            const double width = higherStrikeCall.option.K - lowerStrikeCall.option.K;
            const double bullDebit = lowerStrikeCall.premium - higherStrikeCall.premium;
            const double bearCredit = bullDebit;

            PrintVerticalSpread("Bull call spread",
                                lowerStrikeCall,
                                higherStrikeCall,
                                width - bullDebit,
                                bullDebit,
                                lowerStrikeCall.option.K + bullDebit);
            PrintVerticalSpread("Bear call spread",
                                higherStrikeCall,
                                lowerStrikeCall,
                                bearCredit,
                                width - bearCredit,
                                lowerStrikeCall.option.K + bearCredit);
            return 0;
      }

      if (strategyLeg1.option.IsPut() && strategyLeg2.option.IsPut()) {
            const StrategyLeg& lowerStrikePut = (strategyLeg1.option.K <= strategyLeg2.option.K) ? strategyLeg1 : strategyLeg2;
            const StrategyLeg& higherStrikePut = (strategyLeg1.option.K <= strategyLeg2.option.K) ? strategyLeg2 : strategyLeg1;
            const double width = higherStrikePut.option.K - lowerStrikePut.option.K;
            const double bearDebit = higherStrikePut.premium - lowerStrikePut.premium;
            const double bullCredit = bearDebit;

            PrintVerticalSpread("Bull put spread",
                                lowerStrikePut,
                                higherStrikePut,
                                bullCredit,
                                width - bullCredit,
                                higherStrikePut.option.K - bullCredit);
            PrintVerticalSpread("Bear put spread",
                                higherStrikePut,
                                lowerStrikePut,
                                width - bearDebit,
                                bearDebit,
                                higherStrikePut.option.K - bearDebit);
            return 0;
      }

      const StrategyLeg& callLeg = strategyLeg1.option.IsCall() ? strategyLeg1 : strategyLeg2;
      const StrategyLeg& putLeg = strategyLeg1.option.IsPut() ? strategyLeg1 : strategyLeg2;
      if (absoluteDifference(callLeg.option.K, putLeg.option.K) < 1e-9) {
            PrintStraddleOrStrangle("Straddle", callLeg, putLeg);
      } else {
            PrintStraddleOrStrangle("Strangle", callLeg, putLeg);
      }

      return 0;
}
