#include "EuropeanOption.hpp" 
#include <iostream>
#include <string>
#include <sstream>
#include <math.h>
#include <cmath>
#include <algorithm>

using namespace std;

double test(EuropeanOption& option1, EuropeanOption& option2, int c){
  double deltatotal=abs(option1.Delta())+abs(option2.Delta());
 //cout << '\n' << deltatotal << '\n';
  double sigtotal=option1.sig*(abs(option1.Delta())/deltatotal)+option2.sig*(abs(option1.Delta())/deltatotal);
  double change=sigtotal*option1.U*sqrt(option1.T);
 // cout << '\n' << change << '\n';
  EuropeanOption Option1a=option1;
  EuropeanOption Option2a=option2;
  if(c==0){
  Option1a.U-=change*2;
  Option2a.U-=change*2;
  if(Option1a.U<0){
    Option1a.U=0;
    Option2a.U=0;
  }
  Option1a.T=0;
  Option2a.T=0;
  }
  if(c==1){
  Option1a.U+=change*2;
  Option2a.U+=change*2;
  Option1a.T=0;
  Option2a.T=0;
  }
  return Option1a.Price()-Option2a.Price();
}

double test2(EuropeanOption& option1, EuropeanOption& option2, double premium){
  double deltatotal=abs(option1.Delta())+abs(option2.Delta());
  double sigtotal=option1.sig*(abs(option1.Delta())/deltatotal)+option2.sig*(abs(option1.Delta())/deltatotal);
  double change=sigtotal*option1.U*sqrt(option1.T);
  EuropeanOption Option1a=option1;
  EuropeanOption Option2a=option2;
  
  double a=(option1.K-option1.U+premium)/change;
  double b=(option2.K-option2.U-premium)/change;
  return 1-N(a)+N(b);
}


int main(int argc, char* argv[]){ // All options are European
if (argc != 15 || stod(argv[1])!=stod(argv[8])|| stod(argv[4])!=stod(argv[11])) {
        cout << "Please enter: " << argv[0] << "spot price, strike, volatility, years to expiry, the stock's dividend yield, and the current risk-free interest rate. " << '\n' << "Enter parameters for both options." << endl;
        cout << "The time to expiry and the underlying price must be the same for both options." << endl;
        return 1;
    }
        double spot1 = stod(argv[1]);
        double strike1 = stod(argv[2]);
        double volatility1 = stod(argv[3]);
        double expiry1 = stod(argv[4]);
        double dividend1 = stod(argv[5]);
        double rate1 = stod(argv[6]);
        string type1 = argv[7];
        double spot2 = stod(argv[8]);
        double strike2 = stod(argv[9]);
        double volatility2 = stod(argv[10]);
        double expiry2 = stod(argv[11]);
        double dividend2 = stod(argv[12]);
        double rate2 = stod(argv[13]);
        string type2 = argv[14];
      // Call option on a stock
      EuropeanOption Option1;
      Option1.optType = type1;
      Option1.U = spot1;
      Option1.K = strike1;
      Option1.T = expiry1;
      Option1.r = rate1;
      Option1.sig = volatility1;
    double q = dividend1;
        Option1.b = Option1.r - q;

      EuropeanOption Option2;
      Option2.optType = type2;
      Option2.U = spot2;
      Option2.K = strike2;
      Option2.T = expiry2;
      Option2.r = rate2;
      Option2.sig = volatility2;
      q = dividend2;
      Option2.b = Option2.r - q;

      if(Option1.optType==Option2.optType && Option1.optType=="C"){
        cout << "Option 1 Price: " << Option1.Price() << endl;
        cout << "Option 2 Price: " << Option2.Price() << endl;
        cout << "Bull call spread: " << '\n';
        double proc=(min(Option1.Price(), Option2.Price()))-(max(Option1.Price(), Option2.Price()));
        cout << "Net Proceeds: " << proc << '\n';
        if(Option1.Price()>Option2.Price()){
        cout << "Net Option Delta: " <<Option1.Delta()-Option2.Delta() << std::endl;
        cout << "Net Option Gamma: " << Option1.getGamma()-Option2.getGamma() << std::endl;
        cout << "Net Option Theta: " << Option1.Theta()-Option2.Theta() << std::endl;
        cout << "Net Option Vega: " << Option1.getVega()-Option2.getVega() << std::endl;
        cout << "Worst Case Scenario: " << test(Option1,Option2,0)+proc << endl;
        cout << "Best Case Scenario: " << test(Option1,Option2,1)+proc << endl;

        }
        if(Option2.Price()>Option1.Price()){
        cout << "Net Option Delta: " <<Option2.Delta()-Option1.Delta() << std::endl;
        cout << "Net Option Gamma: " << Option2.getGamma()-Option1.getGamma() << std::endl;
        cout << "Net Option Theta: " << Option2.Theta()-Option1.Theta() << std::endl;
        cout << "Net Option Vega: " << Option2.getVega()+Option1.getVega() << std::endl;
        cout << "Worst Case Scenario: " << test(Option2,Option1,0)+proc << endl;
        cout << "Best Case Scenario: " << test(Option2,Option1,1)+proc << endl;
        }
        cout << "Bear call spread: " << '\n';
        proc=(max(Option1.Price(), Option2.Price()))-(min(Option1.Price(), Option2.Price()));
        cout << "Net Proceeds: " << proc << '\n';
        if(Option2.Price()>Option1.Price()){
        cout << "Net Option Delta: " <<Option1.Delta()-Option2.Delta() << std::endl;
        cout << "Net Option Gamma: " << Option1.getGamma()-Option2.getGamma() << std::endl;
        cout << "Net Option Theta: " << Option1.Theta()-Option2.Theta() << std::endl;
        cout << "Net Option Vega: " << Option1.getVega()-Option2.getVega() << std::endl;
        cout << "Worst Case Scenario: " << proc-test(Option2,Option1,1) << endl;
        cout << "Best Case Scenario: " << proc-test(Option2,Option1,0) << endl;
      }
        if(Option1.Price()>Option2.Price()){
        cout << "Net Option Delta: " <<Option2.Delta()-Option1.Delta() << std::endl;
        cout << "Net Option Gamma: " << Option2.getGamma()-Option1.getGamma() << std::endl;
        cout << "Net Option Theta: " << Option2.Theta()-Option1.Theta() << std::endl;
        cout << "Net Option Vega: " << Option2.getVega()-Option1.getVega() << std::endl;
        cout << "Worst Case Scenario: " << proc-test(Option1,Option2,1) << endl;
        cout << "Best Case Scenario: " << proc-test(Option1,Option2,0) << endl;
        }
      }
      if(Option1.optType==Option2.optType && Option1.optType=="P"){
        cout << "Option 1 Price: " << Option1.Price() << endl;
        cout << "Option 2 Price: " << Option2.Price() << endl;
        cout << "Bull put spread: " << '\n';
        double proc=(max(Option1.Price(), Option2.Price()))-(min(Option1.Price(), Option2.Price()));
        cout << "Net Proceeds: " << proc << '\n';
        if(Option1.Price()>Option2.Price()){
        cout << "Net Option Delta: " <<Option2.Delta()-Option1.Delta() << std::endl;
        cout << "Net Option Gamma: " << Option2.getGamma()-Option1.getGamma() << std::endl;
        cout << "Net Option Theta: " << Option2.Theta()-Option1.Theta() << std::endl;
        cout << "Net Option Vega: " << Option2.getVega()-Option1.getVega() << std::endl;
        cout << "Worst Case Scenario: " << test(Option2,Option1,0)+proc << endl;
        cout << "Best Case Scenario: " << test(Option2,Option1,1)+proc << endl;

        }
        if(Option2.Price()>Option1.Price()){
        cout << "Net Option Delta: " <<Option1.Delta()-Option2.Delta() << std::endl;
        cout << "Net Option Gamma: " << Option1.getGamma()-Option2.getGamma() << std::endl;
        cout << "Net Option Theta: " << Option1.Theta()-Option2.Theta() << std::endl;
        cout << "Net Option Vega: " << Option1.getVega()-Option2.getVega() << std::endl;
        cout << "Worst Case Scenario: " << test(Option1,Option2,0)+proc << endl;
        cout << "Best Case Scenario: " << test(Option1,Option2,1)+proc << endl;
        }
        cout << "Bear put spread: " << '\n';
        proc=(min(Option1.Price(), Option2.Price()))-(max(Option1.Price(), Option2.Price()));
        cout << "Net Proceeds: " << proc << '\n';
        if(Option2.Price()>Option1.Price()){
        cout << "Net Option Delta: " <<Option2.Delta()-Option1.Delta() << std::endl;
        cout << "Net Option Gamma: " << Option2.getGamma()-Option1.getGamma() << std::endl;
        cout << "Net Option Theta: " << Option2.Theta()-Option1.Theta() << std::endl;
        cout << "Net Option Vega: " << Option2.getVega()-Option1.getVega() << std::endl;
        cout << "Worst Case Scenario: " << proc+test(Option2,Option1,1) << endl;
        cout << "Best Case Scenario: " << proc+test(Option2,Option1,0) << endl;
      }
        if(Option1.Price()>Option2.Price()){
        cout << "Net Option Delta: " <<Option1.Delta()-Option2.Delta() << std::endl;
        cout << "Net Option Gamma: " << Option1.getGamma()-Option2.getGamma() << std::endl;
        cout << "Net Option Theta: " << Option1.Theta()-Option2.Theta() << std::endl;
        cout << "Net Option Vega: " << Option1.getVega()-Option2.getVega() << std::endl;
        cout << "Worst Case Scenario: " << test(Option1,Option2,1)+proc << endl;
        cout << "Best Case Scenario: " << test(Option1,Option2,0)+proc << endl;
        }
      }
      if(Option1.optType!=Option2.optType){
        if(Option1.K==Option2.K){
          cout << "Option 1 Price: " << Option1.Price() << endl;
          cout << "Option 2 Price: " << Option2.Price() << endl;
          cout << "Straddle: " << '\n';
          cout << "Net Option Delta: " <<Option1.Delta()+Option2.Delta() << std::endl;
          cout << "Net Option Gamma: " << Option1.getGamma()+Option2.getGamma() << std::endl;
          cout << "Net Option Theta: " << Option1.Theta()+Option2.Theta() << std::endl;
          cout << "Net Option Vega: " << Option1.getVega()+Option2.getVega() << std::endl;
          if(Option1.optType=="C"){
            cout << "Profit Probability: " << 100*test2(Option1, Option2, Option1.Price()+Option2.Price()) << "%";
          }
          else{
            cout << "Profit Probability: " << 100*test2(Option2, Option1, Option1.Price()+Option2.Price()) << "%";
          }
        }
        if(Option1.K!=Option2.K){
          cout << "Option 1 Price: " << Option1.Price() << endl;
          cout << "Option 2 Price: " << Option2.Price() << endl;
          cout << "Strangle: " << '\n';
          cout << "Net Option Delta: " <<Option1.Delta()+Option2.Delta() << std::endl;
          cout << "Net Option Gamma: " << Option1.getGamma()+Option2.getGamma() << std::endl;
          cout << "Net Option Theta: " << Option1.Theta()+Option2.Theta() << std::endl;
          cout << "Net Option Vega: " << Option1.getVega()+Option2.getVega() << std::endl;
         if(Option1.optType=="C"){
            cout << "Profit Probability: " << 100*test2(Option1, Option2, Option1.Price()+Option2.Price()) << "%";
          }
          else{
            cout << "Profit Probability: " << 100*test2(Option2, Option1, Option1.Price()+Option2.Price()) << "%";
          }
        }
      }
}

