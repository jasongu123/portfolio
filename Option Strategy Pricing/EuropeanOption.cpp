#include "EuropeanOption.hpp"
#include <math.h>
#include <cmath>
#include <algorithm>
#include <string>
using std::string;

double N(double x) {
    return 0.5 * (1 + std::erf(x / std::sqrt(2)));
}

double EuropeanOption::CallPrice() const
{
double tmp = sig * sqrt(T);
double d1 = ( log(U/K) + (b+ (sig*sig)*0.5 ) * T )/ tmp; 
double d2 = d1 - tmp;
return (U * exp((b-r)*T) * N(d1)) - (K * exp(-r * T)* N(d2));
}
double EuropeanOption::PutPrice() const
{
double tmp = sig * sqrt(T);
double d1 = ( log(U/K) + (b+ (sig*sig)*0.5 ) * T )/ tmp; 
double d2 = d1 - tmp;
return (K * exp(-r * T)* N(-d2)) - (U * exp((b-r)*T) * N(-d1)); 
}
double EuropeanOption::CallDelta() const{
      double tmp = sig * sqrt(T);
      double d1 = ( log(U/K) + (b+ (sig*sig)*0.5 ) * T )/ tmp; 
      return exp((b-r)*T) * N(d1);

}
double EuropeanOption::PutDelta() const
{
double tmp = sig * sqrt(T);
double d1 = ( log(U/K) + (b+ (sig*sig)*0.5 ) * T )/ tmp; 
return exp((b-r)*T) * (N(d1) - 1.0);
}
double EuropeanOption::Gamma() const{
      double tmp = sig * sqrt(T);
       double d1 = ( log(U/K) + (b+ (sig*sig)*0.5 ) * T )/ tmp; 
       double d2 = d1 - tmp;

       double num=exp(-d1*d1/2)/sqrt(2*M_PI);

       return (exp((b-r)*T)*num)/(U*sig*sqrt(T));
}

double EuropeanOption::CallTheta() const{
      double tmp = sig * sqrt(T);
      double d1 = ( log(U/K) + (b+ (sig*sig)*0.5 ) * T )/ tmp; 
      double d2 = d1 - tmp;
      return ((-U*exp(-d1*d1/2)*sig/(sqrt(2*M_PI)))-r*K*exp(-r*T)*N(d2)+(r-b)*U*exp((b-r)*T)*N(d1))/252;
}

double EuropeanOption::PutTheta() const{
      double tmp = sig * sqrt(T);
      double d1 = ( log(U/K) + (b+ (sig*sig)*0.5 ) * T )/ tmp; 
      double d2 = d1 - tmp;
      return ((-U*exp(-d1*d1/2)*sig/(sqrt(2*M_PI)))+r*K*exp(-r*T)*N(-d2)-(r-b)*U*exp((b-r)*T)*N(-d1))/252;
}

double EuropeanOption::Vega() const{
      double tmp = sig * sqrt(T);
      double d1 = ( log(U/K) + (b+ (sig*sig)*0.5 ) * T )/ tmp; 
      return U*exp(-d1*d1/2)*sqrt(T);
}

void EuropeanOption::init()
{// Initialise all default values
      // Default values
      r = 0.05;
      sig = 0.30;
      K = 65.0;
      T = 0.25;
      U = 60.0;      // U == stock in this case
      b = r;         // Black and Scholes stock option model (1973)
      optType = "C"; // European Call Option (the default type)
}
void EuropeanOption::copy(const EuropeanOption& o2)
{
r = o2.r;
sig = o2.sig;
K = o2.K;
T = o2.T;
U = o2.U;
b = o2.b;
optType = o2.optType;
}
EuropeanOption::EuropeanOption() { // Default call option
init();
}
EuropeanOption::EuropeanOption(const EuropeanOption& o2) 
{ // Copy constructor
copy(o2);
}
EuropeanOption::EuropeanOption (const string& optionType) 
{ // Create option type
init();
optType = optionType;
if (optType == "c")
        optType = "C";
}
EuropeanOption::~EuropeanOption() { // Destructor
}
EuropeanOption& EuropeanOption::operator = (const EuropeanOption& opt2)
{ // Assignment operator (deep copy)
if (this == &opt2) return *this;
          copy (opt2);
          return *this;
}
double EuropeanOption::Price() const
{
      if (optType == "C"){
        return CallPrice();
      }
      else{
        return PutPrice();
      }
}
double EuropeanOption::Delta() const{
if (optType == "C"){
      return CallDelta();
}
else{
      return PutDelta();
}
}

double EuropeanOption::getGamma() const{
      return Gamma();
}

double EuropeanOption::Theta() const{
if (optType == "C"){
      return CallTheta();
}
else{
      return PutTheta();
}
}

double EuropeanOption::getVega() const{
      return Vega();
}

// Modifier functions
void EuropeanOption::toggle(){ // Change option type (C/P, P/C)
if (optType == "C"){
        optType = "P";
}
else{
    optType="C";
}
}