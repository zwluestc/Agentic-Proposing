"""Exact, bounded construction families for the math proposer pilot.

These graders check the declared construction parameters, not natural language.
The independent model committee must additionally check that the question matches
those parameters. All outputs are scalar integer or reduced rational strings.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
import json
import math
import sys


def integer(p, key, lo=-10000, hi=10000):
    value = p[key]
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"{key} must be an integer in [{lo}, {hi}]")
    return value


def int_list(p, key, min_len=1, max_len=30, lo=-10000, hi=10000):
    values = p[key]
    if not isinstance(values,list) or not min_len <= len(values) <= max_len:
        raise ValueError(f"Invalid {key} list length")
    for v in values:
        integer({"value":v},"value",lo,hi)
    return values


def prime(n):
    return n >= 2 and all(n % d for d in range(2,math.isqrt(n)+1))


def factorial_valuation(n,p):
    result = 0
    while n:
        n //= p
        result += n
    return result


def triangle(p):
    points = p["points"]
    if not isinstance(points,list) or len(points) != 3:
        raise ValueError("Require three coordinate points")
    for point in points:
        int_list({"point":point},"point",2,2,-1000,1000)
    a,b,c = points
    cross = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    if not cross:
        raise ValueError("Degenerate triangle")
    return points,cross


def orient(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def segments_intersect(a,b,c,d):
    signs = orient(a,b,c),orient(a,b,d),orient(c,d,a),orient(c,d,b)
    if signs[0]*signs[1] < 0 and signs[2]*signs[3] < 0:
        return True
    for value,point,u,v in [(signs[0],c,a,b),(signs[1],d,a,b),(signs[2],a,c,d),(signs[3],b,c,d)]:
        if value == 0 and min(u[0],v[0]) <= point[0] <= max(u[0],v[0]) and min(u[1],v[1]) <= point[1] <= max(u[1],v[1]):
            return True
    return False


def solve(spec):
    family, p = spec["family"], spec["parameters"]
    if family == "linear":
        a, b, c, d, r, s, u, v = [integer(p, k) for k in ("a", "b", "c", "d", "r", "s", "u", "v")]
        determinant = a*d-b*c
        if determinant == 0:
            raise ValueError("Equations do not have a unique solution")
        x, y = Fraction(r*d-b*s, determinant), Fraction(a*s-r*c, determinant)
        answer = u*x+v*y
    elif family == "quadratic":
        a, b, c = [integer(p, k) for k in ("a", "b", "c")]
        lo, hi = integer(p, "lo", -1000, 1000), integer(p, "hi", -1000, 1000)
        power = integer(p, "power", 1, 4)
        if not a or lo > hi:
            raise ValueError("Require a nonzero quadratic coefficient and ordered interval")
        roots = [x for x in range(lo, hi+1) if a*x*x+b*x+c == 0]
        if not roots:
            raise ValueError("No integer roots in the stated domain")
        answer = sum(x**power for x in roots)
    elif family == "crt":
        moduli, residues = p["moduli"], p["residues"]
        if not 2 <= len(moduli) <= 4 or len(moduli) != len(residues):
            raise ValueError("Require 2-4 residue conditions")
        for m, r in zip(moduli, residues):
            if type(m) is not int or not 2 <= m <= 100 or type(r) is not int or not 0 <= r < m:
                raise ValueError("Invalid modulus/residue")
        lo, hi = integer(p, "lo", 0, 10000), integer(p, "hi", 0, 10000)
        if lo > hi:
            raise ValueError("Reversed interval")
        values = [n for n in range(lo, hi+1) if all(n % m == r for m, r in zip(moduli, residues))]
        target = p["target"]
        if target == "count":
            answer = len(values)
        elif target == "sum":
            answer = sum(values)
        elif target == "minimum" and values:
            answer = min(values)
        else:
            raise ValueError("Invalid target or no minimum exists")
    elif family == "digit_count":
        length = integer(p, "length", 2, 12)
        alphabet = integer(p, "alphabet", 2, 6)
        modulus = integer(p, "modulus", 2, 13)
        residue = integer(p, "residue", 0, modulus-1)
        if type(p.get("leading_zero")) is not bool:
            raise ValueError("leading_zero must be explicit")
        states = {(-1, 0): 1}
        for position in range(length):
            new = {}
            for (previous, total), count in states.items():
                for digit in range(alphabet):
                    if digit == previous or (position == 0 and digit == 0 and not p["leading_zero"]):
                        continue
                    key = digit, (total+digit) % modulus
                    new[key] = new.get(key, 0)+count
            states = new
        answer = sum(count for (_, total), count in states.items() if total == residue)
    elif family == "paths":
        w, h = integer(p, "width", 1, 15), integer(p, "height", 1, 15)
        forbidden = p["forbidden"]
        if not isinstance(forbidden, list) or len(forbidden) > 20:
            raise ValueError("Invalid forbidden lattice points")
        for point in forbidden:
            if not isinstance(point, list) or len(point) != 2 or any(type(n) is not int for n in point) or not (0 <= point[0] <= w and 0 <= point[1] <= h):
                raise ValueError("Forbidden point outside grid")
        blocked = {tuple(point) for point in forbidden}
        table = {}
        for x in range(w+1):
            for y in range(h+1):
                table[x,y] = 0 if (x,y) in blocked else (1 if x == y == 0 else table.get((x-1,y),0)+table.get((x,y-1),0))
        answer = table[w,h]
    elif family == "recurrence":
        n = integer(p, "n", 2, 50)
        a, b, c, f0, f1 = [integer(p,k,-20,20) for k in ("a","b","c","f0","f1")]
        modulus = integer(p, "modulus", 2, 10000)
        for _ in range(2,n+1):
            f0, f1 = f1, (a*f1+b*f0+c) % modulus
        answer = f1 % modulus
    elif family == "probability":
        dice, sides = integer(p,"dice",2,6), integer(p,"sides",2,6)
        threshold = integer(p,"threshold",dice,dice*sides)
        target_sum = integer(p,"target_sum",dice,dice*sides)
        eligible = favorable = 0
        for outcome in product(range(1,sides+1), repeat=dice):
            if sum(outcome) >= threshold:
                eligible += 1
                favorable += sum(outcome) == target_sum
        if not eligible:
            raise ValueError("Condition has probability zero")
        answer = Fraction(favorable, eligible)
    elif family == "optimization":
        bound = integer(p,"bound",2,100)
        a,b,c,d,budget,u,v = [integer(p,k,1,1000) for k in ("a","b","c","d","budget","u","v")]
        feasible = [(x,y) for x in range(bound+1) for y in range(bound+1) if a*x+b*y <= budget and c*x+d*y <= budget]
        answer = max(u*x+v*y for x,y in feasible)
    elif family == "graph":
        n = integer(p,"vertices",3,12)
        edges = p["edges"]
        if not isinstance(edges,list) or not 1 <= len(edges) <= 60 or type(p.get("directed")) is not bool:
            raise ValueError("Invalid graph")
        dist = [[math.inf]*n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for edge in edges:
            if not isinstance(edge,list) or len(edge) != 3:
                raise ValueError("Edges must be [from,to,weight]")
            u,v,w = edge
            if any(type(x) is not int for x in edge) or not (0 <= u < n and 0 <= v < n and 1 <= w <= 100):
                raise ValueError("Invalid edge")
            dist[u][v] = min(dist[u][v],w)
            if not p["directed"]:
                dist[v][u] = min(dist[v][u],w)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i][j] = min(dist[i][j],dist[i][k]+dist[k][j])
        source,target = integer(p,"source",0,n-1),integer(p,"target",0,n-1)
        answer = dist[source][target]
        if not math.isfinite(answer):
            raise ValueError("Target is unreachable")
    elif family == "rational_sum":
        start,end = integer(p,"start",1,1000),integer(p,"end",1,1000)
        shift = integer(p,"shift",1,20)
        scale = integer(p,"scale",1,1000)
        if start > end:
            raise ValueError("Reversed sum range")
        answer = scale*sum((Fraction(1,k*(k+shift)) for k in range(start,end+1)),Fraction(0))
    elif family == "arithmetic_progression":
        a,d = integer(p,"first"),integer(p,"difference")
        start,end = integer(p,"start",1,10000),integer(p,"end",1,10000)
        if start > end:
            raise ValueError("Reversed one-based interval")
        answer = (end-start+1)*(2*a+(start+end-2)*d)//2
    elif family == "geometric_sum":
        a = integer(p,"first")
        numerator = integer(p,"ratio_numerator",-20,20)
        denominator = integer(p,"ratio_denominator",1,20)
        n = integer(p,"terms",1,60)
        q = Fraction(numerator,denominator)
        answer = a*sum((q**i for i in range(n)),Fraction(0))
    elif family == "polynomial_evaluation":
        coefficients = int_list(p,"coefficients",2,12,-1000,1000)
        x = integer(p,"x",-100,100)
        answer = 0
        for coefficient in reversed(coefficients):
            answer = answer*x+coefficient
    elif family == "polynomial_remainder":
        b,c = integer(p,"b",-20,20),integer(p,"c",-20,20)
        exponent = integer(p,"exponent",2,100)
        coefficient = integer(p,"coefficient",0,1)
        constant,linear = 1,0
        for _ in range(exponent):
            constant,linear = -c*linear,constant-b*linear
        answer = [constant,linear][coefficient]
    elif family == "binomial_selection":
        n,m = integer(p,"n",1,100),integer(p,"m",1,100)
        k,l = integer(p,"k",0,n),integer(p,"l",0,m)
        answer = math.comb(n,k)*math.comb(m,l)
    elif family == "bounded_compositions":
        n,cap = integer(p,"variables",1,15),integer(p,"cap",1,30)
        total = integer(p,"total",0,n*cap)
        dp = [1]+[0]*total
        for _ in range(n):
            dp = [sum(dp[t-v] for v in range(min(cap,t)+1)) for t in range(total+1)]
        answer = dp[total]
    elif family == "coin_change":
        coins = int_list(p,"coins",1,12,1,1000)
        if len(coins) != len(set(coins)):
            raise ValueError("Coin denominations must be distinct")
        amount = integer(p,"amount",1,1000)
        dp = [1]+[0]*amount
        for coin in coins:
            for total in range(coin,amount+1):
                dp[total] += dp[total-coin]
        answer = dp[amount]
    elif family == "subset_sum":
        values = int_list(p,"values",1,25,-100,100)
        size = integer(p,"size",0,len(values))
        total = integer(p,"total",-2500,2500)
        dp = {(0,0):1}
        for value in values:
            new = dict(dp)
            for (count,subtotal),ways in dp.items():
                if count < size:
                    key = count+1,subtotal+value
                    new[key] = new.get(key,0)+ways
            dp = new
        answer = dp.get((size,total),0)
    elif family == "derangements":
        n = integer(p,"n",1,80)
        answer = sum((-1)**j*(math.factorial(n)//math.factorial(j)) for j in range(n+1))
    elif family == "involutions":
        n = integer(p,"n",1,80)
        f0,f1 = 1,1
        for k in range(2,n+1):
            f0,f1 = f1,f1+(k-1)*f0
        answer = f1
    elif family == "surjections":
        n = integer(p,"domain_size",1,100)
        m = integer(p,"codomain_size",1,min(n,20))
        answer = sum((-1)**j*math.comb(m,j)*(m-j)**n for j in range(m+1))
    elif family == "catalan":
        n = integer(p,"semilength",1,100)
        answer = math.comb(2*n,n)//(n+1)
    elif family == "necklaces":
        n,colors = integer(p,"length",1,100),integer(p,"colors",2,20)
        answer = sum(colors**math.gcd(n,rotation) for rotation in range(n))//n
    elif family == "ballot":
        a = integer(p,"a_votes",2,100)
        b = integer(p,"b_votes",1,a-1)
        answer = (a-b)*math.comb(a+b,b)//(a+b)
    elif family == "permutation_inversions":
        n = integer(p,"n",2,40)
        k = integer(p,"inversions",0,n*(n-1)//2)
        dp = [1]+[0]*k
        for size in range(1,n+1):
            dp = [sum(dp[t-i] for i in range(min(size-1,t)+1)) for t in range(k+1)]
        answer = dp[k]
    elif family == "gcd_lcm":
        g,l = integer(p,"gcd",1,100000),integer(p,"lcm",1,100000000)
        if l % g:
            raise ValueError("LCM must be a multiple of GCD")
        ratio,distinct = l//g,0
        d = 2
        while d*d <= ratio:
            if ratio % d == 0:
                distinct += 1
                while ratio % d == 0:
                    ratio //= d
            d += 1
        distinct += ratio > 1
        answer = 2**distinct
    elif family == "divisor_sum":
        n = integer(p,"n",1,100000000)
        power = integer(p,"power",1,3)
        answer = 0
        for d in range(1,math.isqrt(n)+1):
            if n % d == 0:
                answer += d**power
                if d*d != n:
                    answer += (n//d)**power
    elif family == "totient":
        n = integer(p,"n",2,1000000000)
        remaining,answer,d = n,n,2
        while d*d <= remaining:
            if remaining % d == 0:
                answer -= answer//d
                while remaining % d == 0:
                    remaining //= d
            d += 1
        if remaining > 1:
            answer -= answer//remaining
    elif family == "modular_power":
        base = integer(p,"base",-1000000000,1000000000)
        exponent = integer(p,"exponent",1,1000000000)
        modulus = integer(p,"modulus",2,1000000000)
        answer = pow(base,exponent,modulus)
    elif family == "multiplicative_order":
        modulus = integer(p,"modulus",2,5000)
        base = integer(p,"base",1,10000)
        if math.gcd(base,modulus) != 1:
            raise ValueError("Multiplicative order requires coprime base and modulus")
        value,answer = 1,0
        while True:
            value,answer = value*base % modulus,answer+1
            if value == 1:
                break
    elif family == "valuation_factorial":
        n = integer(p,"n",1,1000000000)
        p0 = integer(p,"prime",2,10000)
        if not prime(p0):
            raise ValueError("Require a prime factor")
        answer = factorial_valuation(n,p0)
    elif family == "trailing_zeros_binomial":
        n = integer(p,"n",2,10000000)
        k = integer(p,"k",1,n-1)
        answer = min(factorial_valuation(n,prime0)-factorial_valuation(k,prime0)-factorial_valuation(n-k,prime0) for prime0 in (2,5))
    elif family == "diophantine_nonnegative":
        a,b = integer(p,"a",1,1000),integer(p,"b",1,1000)
        total = integer(p,"total",1,100000)
        answer = sum((total-a*x) % b == 0 for x in range(total//a+1))
    elif family == "integer_triangles":
        perimeter = integer(p,"perimeter",3,300)
        answer = sum(1 for a in range(1,perimeter//3+1) for b in range(a,(perimeter-a)//2+1)
                     if a+b > perimeter-a-b)
    elif family == "triangle_area_squared":
        _,cross = triangle(p)
        answer = Fraction(cross*cross,4)
    elif family == "lattice_polygon_interior":
        points = p["vertices"]
        if not isinstance(points,list) or not 3 <= len(points) <= 12:
            raise ValueError("Require 3-12 polygon vertices")
        for point in points:
            int_list({"point":point},"point",2,2,-1000,1000)
        if len({tuple(v) for v in points}) != len(points):
            raise ValueError("Polygon vertices must be distinct; omit repeated closing vertex")
        edges = list(zip(points,points[1:]+points[:1]))
        for i,(a,b) in enumerate(edges):
            # Collinear adjacent edges are rejected to exclude backtracking.
            if orient(a,b,points[(i+2)%len(points)]) == 0:
                raise ValueError("Adjacent polygon edges must be noncollinear")
            for j,(c,d) in enumerate(edges):
                if j > i+1 and not (i == 0 and j == len(edges)-1) and segments_intersect(a,b,c,d):
                    raise ValueError("Polygon must be simple")
        twice_area = abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in edges))
        boundary = sum(math.gcd(abs(a[0]-b[0]),abs(a[1]-b[1])) for a,b in edges)
        answer = (twice_area-boundary+2)//2
        if not twice_area or answer < 0 or (twice_area-boundary) % 2:
            raise ValueError("Invalid lattice polygon")
    elif family == "circumradius_squared":
        points,cross = triangle(p)
        squared_lengths = [(a[0]-b[0])**2+(a[1]-b[1])**2 for a,b in zip(points,points[1:]+points[:1])]
        answer = Fraction(math.prod(squared_lengths),4*cross*cross)
    elif family == "markov_chain":
        matrix = p["transition"]
        if not isinstance(matrix,list) or not 2 <= len(matrix) <= 5:
            raise ValueError("Require 2-5 states")
        n = len(matrix)
        rows = []
        for row in matrix:
            if not isinstance(row,list) or len(row) != n or any(type(v) not in (str,int) or len(str(v)) > 30 for v in row):
                raise ValueError("Require square matrix of exact integer/fraction strings")
            row = [Fraction(v) for v in row]
            if any(v < 0 or v > 1 for v in row) or sum(row) != 1:
                raise ValueError("Transition rows must be nonnegative and sum to one")
            rows.append(row)
        start,target = integer(p,"start",0,n-1),integer(p,"target",0,n-1)
        steps = integer(p,"steps",1,30)
        probabilities = [Fraction(int(i == start)) for i in range(n)]
        for _ in range(steps):
            probabilities = [sum((probabilities[i]*rows[i][j] for i in range(n)),Fraction(0)) for j in range(n)]
        answer = probabilities[target]
    elif family == "urn_hypergeometric":
        red,blue = integer(p,"red",1,100),integer(p,"blue",1,100)
        draws = integer(p,"draws",1,red+blue)
        k = integer(p,"target_red",max(0,draws-blue),min(red,draws))
        answer = Fraction(math.comb(red,k)*math.comb(blue,draws-k),math.comb(red+blue,draws))
    elif family == "expected_max":
        dice,sides = integer(p,"dice",1,20),integer(p,"sides",2,100)
        answer = Fraction(sum(k*(k**dice-(k-1)**dice) for k in range(1,sides+1)),sides**dice)
    else:
        raise ValueError("Unknown math construction family")
    return {"answer": str(answer), "family": family, "verified_parameters": p}


if __name__ == "__main__":
    try:
        spec = json.load(sys.stdin)
        if spec.get("family") != 'subset_sum':
            raise ValueError("Input family does not match this Skill")
        print(json.dumps(solve(spec),ensure_ascii=False))
    except (ValueError,KeyError,TypeError,ZeroDivisionError) as exc:
        print(json.dumps({"error":str(exc)}))
        sys.exit(1)
