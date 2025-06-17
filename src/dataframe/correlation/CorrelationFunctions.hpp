// Qn Tools
//
// Copyright (C) 2020  Lukas Kreis Ilya Selyuzhenkov
// Contact: l.kreis@gsi.de; ilya.selyuzhenkov@gmail.com
// For a full list of contributors please see docs/Credits
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
#ifndef QNTOOLS_CORRELATIONFUNCTIONS_H_
#define QNTOOLS_CORRELATIONFUNCTIONS_H_

#include <cmath>
#include <complex>

namespace Qn::Correlation::TwoParticle {
inline auto xx(unsigned int h_a, unsigned int h_b) {
  return [h_a, h_b](const Qn::QVector &a, const Qn::QVector &b) {
    return a.x(h_a) * b.x(h_b);
  };
}
inline auto yy(unsigned int h_a, unsigned int h_b) {
  return [h_a, h_b](const Qn::QVector &a, const Qn::QVector &b) {
    return a.y(h_a) * b.y(h_b);
  };
}
inline auto yx(unsigned int h_a, unsigned int h_b) {
  return [h_a, h_b](const Qn::QVector &a, const Qn::QVector &b) {
    return a.y(h_a) * b.x(h_b);
  };
}
inline auto xy(unsigned int h_a, unsigned int h_b) {
  return [h_a, h_b](const Qn::QVector &a, const Qn::QVector &b) {
    return a.x(h_a) * b.y(h_b);
  };
}
inline auto ScalarProduct(unsigned int h_u, unsigned int h_Q) {
  return [h_u, h_Q](const Qn::QVector &u, const Qn::QVector &Q) {
    return u.x(h_u) * Q.x(h_Q) + u.y(h_u) * Q.y(h_Q);
  };
}

inline auto c2(unsigned int h_u) {
  return [h_u](const Qn::QVector &u) {
    double ret = 0.;
    auto Q = u.DeNormal();
    auto m = Q.sumweights();
    if (m < 2.) {
      ret = 0.;
    } else {
      ret = (ScalarProduct(Q, Q, h_u) - m) / (m * (m - 1.));
    }
    return ret;
  };
}
  
// Weighted v2{2} method.
// Input Q-vectors:
//   Q11 = sum w^1 * un
//   Q12 = sum w^2 * un
// Q11 and Q12 are the same except weight
inline auto wc2(unsigned int h_u) {
  return [h_u](const Qn::QVector &q11, const Qn::QVector &q12) {
    double ret = 0.;
    auto Q11 = q11.DeNormal();
    auto Q12 = q12.DeNormal();
    auto S12 = Q12.sumweights();
    auto S11 = Q11.sumweights();
    auto S21 = S11 * S11;
    auto denom = S21 - S12;
    if (denom == 0.) {
      ret = 0.;
    } else {
      ret = (ScalarProduct(Q11, Q11, h_u) - S12) / denom;
    }
    return ret;
  };
} 

// v2{2} method
// p = sum 1. * un - POI
// r = sum 1. * un - RFP
// q = sum 1. * un - both POI & RFP
inline auto d2(unsigned int h_u) {
  return [h_u](const Qn::QVector &p, const Qn::QVector &r, const Qn::QVector &q) {
    double ret = 0.;
    auto mp = p.sumweights();
    auto mr = r.sumweights();
    if (mp < 2. || mr < 2.) {
      return ret = 0.;
    } else {
      auto P = p.DeNormal();
      auto R = r.DeNormal();
      auto Q = q.DeNormal();
      std::complex<double> p0{P.sumweights(), 0};
      std::complex<double> r0{R.sumweights(), 0};
      std::complex<double> q0{Q.sumweights(), 0};
      std::complex<double> p1{P.x(h_u), P.y(h_u)};
      std::complex<double> r1{R.x(h_u), R.y(h_u)};
      ret = (p1 * std::conj(r1) - q0).real() / (p0 * r0 - q0).real();
    }
    return ret;
  };
}

// Weighted v2{2} method
// Input Q-vectors:
//   p = sum 1. * un - POI
//   r = sum w^1 * un - RFP
//   q = sum w^1 * un  - both RFP and POI
inline auto wd2(unsigned int h_u) {
  return [h_u](const Qn::QVector &p, const Qn::QVector &r, const Qn::QVector &q) {
    double ret = 0.;
    auto mp = p.sumweights();
    auto mr = r.sumweights();
    if (mp < 2. || mr < 2.) {
      return 0.;
    } else {
      auto P = p.DeNormal();
      auto R = r.DeNormal();
      auto Q = q.DeNormal();
      std::complex<double> p0{P.sumweights(), 0};
      std::complex<double> r0{R.sumweights(), 0};
      std::complex<double> q0{Q.sumweights(), 0};
      std::complex<double> p1{P.x(h_u), P.y(h_u)};
      std::complex<double> r1{R.x(h_u), R.y(h_u)};
      ret = (p1 * std::conj(r1) - q0).real() / (p0 * r0 - q0).real();
    }
    return ret;
  };
}

inline auto n2() {
  return [](const Qn::QVector &q) {
    auto m = q.sumweights();
    return m * (m - 1);
  };
}

inline auto nd2() {
  return [](const Qn::QVector &p, const Qn::QVector &r) {
    auto mp = p.sumweights();
    auto mr = r.sumweights();
    return mp * mr - mp;
  };
}

}  // namespace Qn::Correlation::TwoParticle

/**
 * mixed harmonics
 */
namespace Qn::Correlation::MixedHarmonics {
inline auto xxx(unsigned int h_a, unsigned int h_b, unsigned int h_c) {
  return [h_a, h_b, h_c](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc) {
    return u.x(h_a) * Qb.x(h_b) * Qc.x(h_c);
  };
}
inline auto xyy(unsigned int h_a, unsigned int h_b, unsigned int h_c) {
  return [h_a, h_b, h_c](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc) {
    return u.x(h_a) * Qb.y(h_b) * Qc.y(h_c);
  };
}
inline auto yxy(unsigned int h_a, unsigned int h_b, unsigned int h_c) {
  return [h_a, h_b, h_c](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc) {
    return u.y(h_a) * Qb.x(h_b) * Qc.y(h_c);
  };
}
inline auto yyx(unsigned int h_a, unsigned int h_b, unsigned int h_c) {
  return [h_a, h_b, h_c](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc) {
    return u.y(h_a) * Qb.y(h_b) * Qc.x(h_c);
  };
}
inline auto yyy(unsigned int h_a, unsigned int h_b, unsigned int h_c) {
  return [h_a, h_b, h_c](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc) {
    return u.y(h_a) * Qb.y(h_b) * Qc.y(h_c);
  };
}
inline auto xyx(unsigned int h_a, unsigned int h_b, unsigned int h_c) {
  return [h_a, h_b, h_c](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc) {
    return u.x(h_a) * Qb.y(h_b) * Qc.x(h_c);
  };
}
inline auto yxx(unsigned int h_a, unsigned int h_b, unsigned int h_c) {
  return [h_a, h_b, h_c](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc) {
    return u.y(h_a) * Qb.x(h_b) * Qc.x(h_c);
  };
}
inline auto xxy(unsigned int h_a, unsigned int h_b, unsigned int h_c) {
  return [h_a, h_b, h_c](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc) {
    return u.x(h_a) * Qb.x(h_b) * Qc.y(h_c);
  };
}
inline auto ScalarProduct(unsigned int h_a, unsigned int h_b, unsigned int h_c) {
  return [h_a, h_b, h_c](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc) {
    return u.x(h_a) * Qb.x(h_b) * Qc.x(h_c) - u.x(h_a) * Qb.y(h_b) * Qc.y(h_c) + 
      u.y(h_a) * Qb.y(h_b) * Qc.x(h_c) + u.y(h_a) * Qb.x(h_b) * Qc.y(h_c);
  };
}
}  // namespace Qn::Correlation::MixedHarmonics

/**
 * Four particle correlation functions
 */
namespace Qn::Correlation::FourParticle {

inline auto xxxx(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.x(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.x(h_d);
  };
}

inline auto yxxx(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.y(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.x(h_d);
  };
}

inline auto xyxx(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.x(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.x(h_d);
  };
}

inline auto yyxx(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.y(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.x(h_d);
  };
}

inline auto xxyx(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.x(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.x(h_d);
  };
}

inline auto yxyx(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.y(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.x(h_d);
  };
}

inline auto xyyx(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.x(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.x(h_d);
  };
}

inline auto yyyx(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.y(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.x(h_d);
  };
}

inline auto xxxy(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.x(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.y(h_d);
  };
}

inline auto yxxy(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.y(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.y(h_d);
  };
}

inline auto xyxy(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.x(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.y(h_d);
  };
}

inline auto yyxy(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.y(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.y(h_d);
  };
}

inline auto xxyy(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.x(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.y(h_d);
  };
}

inline auto yxyy(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.y(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.y(h_d);
  };
}

inline auto xyyy(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.x(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.y(h_d);
  };
}

inline auto yyyy(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.y(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.y(h_d);
  };
}

// // < u Q Q* Q* > scalar product - w/o mixed harmonics
// inline auto ScalarProductNoMixed(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
//   return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
//                          const Qn::QVector &Qc, const Qn::QVector &Qd) {
//     return u.x(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.x(h_d) - u.x(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.y(h_d) + 
//       u.x(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.y(h_d) + u.x(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.x(h_d) +
//       u.y(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.y(h_d) + u.y(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.x(h_d) -
//       u.y(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.x(h_d) + u.y(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.y(h_d);
//   };
// }

// < u Q* Q* Q* > scalar product - w/ mixed harmonics
inline auto ScalarProduct(unsigned int h_a, unsigned int h_b, unsigned int h_c, unsigned int h_d) {
  return [h_a, h_b, h_c, h_d](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd) {
    return u.x(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.x(h_d) - u.x(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.y(h_d) - 
      u.x(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.y(h_d) - u.x(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.x(h_d) +
      u.y(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.y(h_d) + u.y(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.x(h_d) +
      u.y(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.x(h_d) - u.y(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.y(h_d);
  };
}
  
inline auto n4() {
  return [](const Qn::QVector &q) {
    auto m = q.sumweights();
    return m * (m - 1.) * (m - 2.) * (m - 3.);
  };
}

inline auto nd4() {
  return [](const Qn::QVector &p, const Qn::QVector &r) {
    auto mp = p.sumweights();
    auto mr = r.sumweights();
    return (mp * mr - 3. * mp) * (mr - 1.) * (mr - 2.);
  };
}

// v2{4} method
// p = sum 1. * un - POI
// r = sum 1. * un - RFP
// q = sum 1. * un - both POI & RFP
inline auto d4(unsigned int h_u) {
  return [h_u](const Qn::QVector &p, const Qn::QVector &r, const Qn::QVector &q) {
    const auto R = r.DeNormal();
    const auto RM = r.sumweights();
    const auto P = p.DeNormal();
    const auto PM = p.sumweights();
    const auto Q = q.DeNormal();
    const auto QM = q.sumweights();
    if (PM < 4.) return 0.;
    const auto p0 = std::complex<double>{PM, 0.};
    const auto r0 = std::complex<double>{RM, 0.};
    const auto q0 = std::complex<double>{QM, 0.};
    const auto p1 = std::complex<double>{P.x(h_u), P.y(h_u)};
    const auto r1 = std::complex<double>{R.x(h_u), R.y(h_u)};
    const auto r2 = std::complex<double>{R.x(2 * h_u), R.y(2 * h_u)};
    const auto q1 = std::complex<double>{Q.x(h_u), Q.y(h_u)};
    const auto q2 = std::complex<double>{Q.x(2 * h_u), Q.y(2 * h_u)};

    auto c4 = p1 * r1 * std::conj(r1) * std::conj(r1) -
              q2 * std::conj(r1) * std::conj(r1) - p1 * r1 * std::conj(r2) -
              2. * r0 * p1 * std::conj(r1) - 2. * q0 * r1 * std::conj(r1) +
              7. * q1 * std::conj(r1) - r1 * std::conj(q1) + q2 * std::conj(r2) +
              2. * p1 * std::conj(r1) + 2. * q0 * r0 - 6. * q0;
      
    return c4.real() / ((p0 * r0 - 3. * q0) * (r0 - 1.) * (r0 - 2.)).real();
  };
};

// Weighted v2{4} method
// p10 = sum 1. * un - POI
// r11 = sum w^1 * un - RFP
// r22 = sum w^2 * un - RFP (for Q2n)
// r13 = sum w^3 * un - RFP
// r14 = sum w^4 * un - RFP
// q21 = sum w^1 * un - both POI & RFP (for q2n)
// q12 = sum w^2 * un - both POI & RFP
// q13 = sum w^3 * un - both POI & RFP
inline auto wd4(unsigned int h_u) {
  return [h_u](const Qn::QVector &p10, const Qn::QVector &q21, const Qn::QVector &q12, const Qn::QVector &q13,
               const Qn::QVector &r11, const Qn::QVector &r22, const Qn::QVector &r13, const Qn::QVector &r14) {
    const auto P10 = p10.DeNormal();
    const auto R11 = r11.DeNormal();
    const auto R22 = r22.DeNormal();
    const auto R13 = r13.DeNormal();
    const auto R14 = r14.DeNormal();
    const auto Q21 = q21.DeNormal();
    const auto Q12 = q12.DeNormal();
    const auto Q13 = q13.DeNormal();
    const auto S11 = std::complex<double>{R11.sumweights(), 0.};
    const auto S12 = std::complex<double>{R22.sumweights(), 0.};
    const auto S13 = std::complex<double>{R13.sumweights(), 0.};
    const auto S14 = std::complex<double>{R14.sumweights(), 0.};
    const auto S21 = S11 * S11;
    const auto S31 = S11 * S21;
    const auto mp  = std::complex<double>{P10.sumweights(), 0.};
    const auto s11 = std::complex<double>{Q21.sumweights(), 0.};
    const auto s12 = std::complex<double>{Q12.sumweights(), 0.};
    const auto s13 = std::complex<double>{Q13.sumweights(), 0.};
    const auto denom = mp*(S31 - 3.*S11*S12 + 2.*S13) - 3.*(s11*(S21-S12)+2.*(s13-s12*S11));
    if (denom.real() == 0) return 0.;

    const auto fp10 = std::complex<double>{P10.x(h_u), P10.y(h_u)};
    const auto fq21 = std::complex<double>{Q21.x(2*h_u), Q21.y(2*h_u)};
    const auto fq12 = std::complex<double>{Q12.x(h_u), Q12.y(h_u)};
    const auto fq13 = std::complex<double>{Q13.x(h_u), Q13.y(h_u)};
    const auto fr11 = std::complex<double>{R11.x(h_u), R11.y(h_u)};
    const auto fr22 = std::complex<double>{R22.x(2*h_u), R22.y(2*h_u)};
    const auto fr13 = std::complex<double>{R13.x(h_u), R13.y(h_u)};
    const auto fr14 = std::complex<double>{R14.x(h_u), R14.y(h_u)};

    auto c4 = fp10 * fr11 * std::conj(fr11) * std::conj(fr11) -
              fq21 * std::conj(fr11) * std::conj(fr11) - fp10 * fr11 * std::conj(fr22) -
              2. * S12 * fp10 * std::conj(fr11) - 2. * s11 * fr11 * std::conj(fr11) +
              7. * fq12 * std::conj(fr11) - fr11 * std::conj(fq12) + fq21 * std::conj(fr22) +
              2. * fp10 * std::conj(fr13) + 2. * s11 * S12 - 6. * s13;
      
    return c4.real() / denom.real();
  };
};

inline auto c4(unsigned int h_u) {
  return [h_u](const Qn::QVector &u) {
    float ret = 0.;
    auto Q = u.DeNormal();
    auto M = u.sumweights();
    if (M < 4.) {
      ret = 0.;
    } else {
      auto x = Q.x(h_u);
      auto y = Q.y(h_u);
      auto x2 = Q.x(2 * h_u);
      auto y2 = Q.y(2 * h_u);
      auto Q_mag = std::sqrt(x * x + y * y);
      auto Q_2n_mag = std::sqrt(x2 * x2 + y2 * y2);
      auto real = x2 * x * x - x2 * y * y + y2 * y * x + y2 * x * y;
      auto term_1 =
          (Q_mag * Q_mag * Q_mag * Q_mag + Q_2n_mag * Q_2n_mag - 2 * real) /
          (M * (M - 1) * (M - 2) * (M - 3));
      auto term_2 = (2 * (M - 2) * Q_mag * Q_mag - M * (M - 3)) /
                    (M * (M - 1) * (M - 2) * (M - 3));
      ret = term_1 - 2 * term_2;
    }
    return ret;
  };
}

// Weighted v2{4} method.
// Input Q-vectors:
//   Q11 = sum w^1 * un
//   Q22 = sum w^2 * un
//   Q13 = sum w^3 * un
//   Q14 = sum w^4 * un
// Q11, Q22, Q13, Q14 are the same except weights
inline auto wc4(unsigned int h_u) {
  return [h_u](const Qn::QVector &Q11, const Qn::QVector &Q22, const Qn::QVector &Q13, const Qn::QVector &Q14) {
    double ret = 0.;
    auto fQ11 = Q11.DeNormal();
    auto fQ22 = Q22.DeNormal();
    auto fQ13 = Q11.DeNormal();
    auto fQ14 = Q14.DeNormal();
    auto Q11_mag = std::sqrt(fQ11.x(h_u)*fQ11.x(h_u) + fQ11.y(h_u)*fQ11.y(h_u));
    auto Q11_mag2 = Q11_mag * Q11_mag;
    auto Q11_mag4 = Q11_mag2 * Q11_mag2;
    auto Q22_mag = std::sqrt(fQ22.x(2*h_u)*fQ22.x(2*h_u) + fQ22.y(2*h_u)*fQ22.y(2*h_u));
    auto Q22_mag2 = Q22_mag * Q22_mag;
    auto S11 = fQ11.sumweights();
    auto S12 = fQ22.sumweights();
    auto S13 = fQ13.sumweights();
    auto S14 = fQ14.sumweights();
    auto S21 = S11 * S11;
    auto S41 = S21 * S21;
    auto S22 = S12 * S12;
    auto denom = S41 - 6.*S12*S21 + 8.*S13*S11 + 3.*S22 - 6.*S14;
    if (denom == 0.) {
      ret = 0.;
    } else {
      auto scalar_term = (Q11_mag4 + Q22_mag2 - 4.*S12*Q11_mag2) / denom;
      auto re1 = 8. * ScalarProduct(fQ13, fQ11, h_u) / denom;
      auto re2 = 2. * (fQ22.x(2*h_u)*fQ11.x(h_u)*fQ11.x(h_u) - fQ22.x(2*h_u)*fQ11.y(h_u)*fQ11.y(h_u) +
                       fQ22.y(2*h_u)*fQ11.y(h_u)*fQ11.x(h_u) + fQ22.y(2*h_u)*fQ11.x(h_u)*fQ11.y(h_u)) / denom;
      auto real_term = re1 - re2;
      auto weight_term = (2.*S22 + 6.*S14) / denom;
      ret = scalar_term + real_term - weight_term;
    }
    return ret;
  };
}

}  // namespace Qn::Correlation::FourParticle

/**
 * Five particle correlation functions
 */
namespace Qn::Correlation::FiveParticle {

inline auto xxxxx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.x(h_d) * Qk.x(h_k);
  };
}

inline auto yxxxx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.x(h_d) * Qk.x(h_k);
  };
}

inline auto xyxxx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.x(h_d) * Qk.x(h_k);
  };
}

inline auto yyxxx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.x(h_d) * Qk.x(h_k);
  };
}

inline auto xxyxx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.x(h_d) * Qk.x(h_k);
  };
}

inline auto yxyxx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.x(h_d) * Qk.x(h_k);
  };
}

inline auto xyyxx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.x(h_d) * Qk.x(h_k);
  };
}

inline auto yyyxx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.x(h_d) * Qk.x(h_k);
  };
}

inline auto xxxyx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.y(h_d) * Qk.x(h_k);
  };
}

inline auto yxxyx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.y(h_d) * Qk.x(h_k);
  };
}

inline auto xyxyx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.y(h_d) * Qk.x(h_k);
  };
}

inline auto yyxyx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.y(h_d) * Qk.x(h_k);
  };
}

inline auto xxyyx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.y(h_d) * Qk.x(h_k);
  };
}

inline auto yxyyx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.y(h_d) * Qk.x(h_k);
  };
}

inline auto xyyyx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.y(h_d) * Qk.x(h_k);
  };
}

inline auto yyyyx(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.y(h_d) * Qk.x(h_k);
  };
}

inline auto xxxxy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.x(h_d) * Qk.y(h_k);
  };
}

inline auto yxxxy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.x(h_d) * Qk.y(h_k);
  };
}

inline auto xyxxy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.x(h_d) * Qk.y(h_k);
  };
}

inline auto yyxxy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.x(h_d) * Qk.y(h_k);
  };
}

inline auto xxyxy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.x(h_d) * Qk.y(h_k);
  };
}

inline auto yxyxy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.x(h_d) * Qk.y(h_k);
  };
}

inline auto xyyxy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.x(h_d) * Qk.y(h_k);
  };
}

inline auto yyyxy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.x(h_d) * Qk.y(h_k);
  };
}

inline auto xxxyy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.y(h_d) * Qk.y(h_k);
  };
}

inline auto yxxyy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.y(h_d) * Qk.y(h_k);
  };
}

inline auto xyxyy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.y(h_d) * Qk.y(h_k);
  };
}

inline auto yyxyy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.y(h_d) * Qk.y(h_k);
  };
}

inline auto xxyyy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.y(h_d) * Qk.y(h_k);
  };
}

inline auto yxyyy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.y(h_d) * Qk.y(h_k);
  };
}

inline auto xyyyy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.y(h_d) * Qk.y(h_k);
  };
}

inline auto yyyyy(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                    unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.y(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.y(h_d) * Qk.y(h_k);
  };
}

// < u Q* Q* Q* Q* > scalar product - w/ mixed harmonics
inline auto ScalarProduct(unsigned int h_a, unsigned int h_b, unsigned int h_c,
                          unsigned int h_d, unsigned int h_k) {
  return [h_a, h_b, h_c, h_d, h_k](const Qn::QVector &u, const Qn::QVector &Qb,
                         const Qn::QVector &Qc, const Qn::QVector &Qd,
                         const Qn::QVector &Qk) {
    return u.x(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.x(h_d) * Qk.x(h_k) - u.x(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.y(h_d) * Qk.y(h_k) - 
      u.x(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.x(h_d) * Qk.y(h_k) - u.x(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.y(h_d) * Qk.x(h_k) -
      u.x(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.x(h_d) * Qk.y(h_k) - u.x(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.y(h_d) * Qk.x(h_k) -
      u.x(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.x(h_d) * Qk.x(h_k) + u.x(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.y(h_d) * Qk.y(h_k) +
      u.y(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.x(h_d) * Qk.y(h_k) + u.y(h_a) * Qb.x(h_b) * Qc.x(h_c) * Qd.y(h_d) * Qk.x(h_k) +
      u.y(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.x(h_d) * Qk.x(h_k) - u.y(h_a) * Qb.x(h_b) * Qc.y(h_c) * Qd.y(h_d) * Qk.y(h_k) +
      u.y(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.x(h_d) * Qk.x(h_k) - u.y(h_a) * Qb.y(h_b) * Qc.x(h_c) * Qd.y(h_d) * Qk.y(h_k) -
      u.y(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.x(h_d) * Qk.y(h_k) - u.y(h_a) * Qb.y(h_b) * Qc.y(h_c) * Qd.y(h_d) * Qk.x(h_k);
  };
}
}  // namespace Qn::Correlation::FiveParticle
#endif  // QNTOOLS_CORRELATIONFUNCTIONS_H_
