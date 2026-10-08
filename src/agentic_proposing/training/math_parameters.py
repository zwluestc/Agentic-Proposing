"""Deterministic diverse parameter suggestions, with no supplied gold answers."""
import random
from fractions import Fraction


def suggest(family, index):
    r = random.Random(f"math-proposer:{family}:{index}")
    if family == "linear":
        a,b,c,d = r.randint(2,9),r.randint(1,7),r.randint(1,7),-r.randint(1,9)
        x,y = r.randint(2,30),r.randint(2,30)
        p = dict(a=a,b=b,c=c,d=d,r=a*x+b*y,s=c*x+d*y,u=r.randint(1,8),v=r.randint(1,8))
    elif family == "quadratic":
        x,y = r.sample(range(-20,31),2)
        a = r.randint(1,5)
        p = dict(a=a,b=-a*(x+y),c=a*x*y,lo=min(x,y)-r.randint(1,8),hi=max(x,y)+r.randint(1,8),power=r.randint(2,4))
    elif family == "crt":
        moduli = r.sample([3,4,5,7,8,9,11,13,17],3)
        x = r.randint(20,500)
        p = dict(moduli=moduli,residues=[x%m for m in moduli],lo=1,hi=r.randint(1000,5000),target=r.choice(["count","sum","minimum"]))
    elif family == "digit_count":
        p = dict(length=r.randint(3,7),alphabet=r.randint(3,6),modulus=r.randint(3,11),residue=r.randint(0,2),leading_zero=r.choice([True,False]))
    elif family == "paths":
        w,h = r.randint(4,12),r.randint(4,12)
        points = r.sample([(x,y) for x in range(1,w) for y in range(1,h)],r.randint(1,3))
        p = dict(width=w,height=h,forbidden=[list(x) for x in points])
    elif family == "recurrence":
        p = dict(n=r.randint(8,35),a=r.randint(1,4),b=r.randint(1,3),c=r.randint(0,6),f0=r.randint(0,5),f1=r.randint(1,9),modulus=r.choice([97,101,127,256,997,1000]))
    elif family == "probability":
        dice,sides = r.randint(2,4),r.randint(3,6)
        threshold = r.randint(dice+2,dice*sides-2)
        p = dict(dice=dice,sides=sides,threshold=threshold,target_sum=r.randint(threshold,dice*sides))
    elif family == "optimization":
        p = dict(bound=r.randint(10,30),a=r.randint(2,9),b=r.randint(2,9),c=r.randint(2,9),d=r.randint(2,9),budget=r.randint(40,180),u=r.randint(3,12),v=r.randint(3,12))
    elif family == "graph":
        n = r.randint(5,8)
        edges = [[i,i+1,r.randint(2,18)] for i in range(n-1)]
        edges += [[i,j,r.randint(4,30)] for i,j in r.sample([(i,j) for i in range(n) for j in range(i+2,n)],4)]
        p = dict(vertices=n,edges=edges,directed=r.choice([True,False]),source=0,target=n-1)
    elif family == "rational_sum":
        start = r.randint(1,25)
        p = dict(start=start,end=start+r.randint(10,70),shift=r.randint(2,8),scale=r.randint(2,20))
    elif family == "arithmetic_progression":
        start = r.randint(2,20)
        p = dict(first=r.randint(-30,40),difference=r.choice([-7,-3,2,5,9]),start=start,end=start+r.randint(10,90))
    elif family == "geometric_sum":
        p = dict(first=r.randint(2,20),ratio_numerator=r.choice([-5,-3,2,3,5,7]),ratio_denominator=r.choice([2,3,4,5]),terms=r.randint(4,15))
    elif family == "polynomial_evaluation":
        p = dict(coefficients=[r.randint(-12,12) for _ in range(r.randint(4,8))]+[r.randint(1,9)],x=r.choice([-7,-5,-3,2,3,5]))
    elif family == "polynomial_remainder":
        p = dict(b=r.randint(-7,7),c=r.choice([-5,-3,-2,1,2,3,5]),exponent=5+index%89,coefficient=r.randint(0,1))
    elif family == "binomial_selection":
        n,m = r.randint(10,30),r.randint(8,25)
        p = dict(n=n,m=m,k=r.randint(3,n-2),l=r.randint(2,m-2))
    elif family == "bounded_compositions":
        variables,cap = r.randint(4,10),r.randint(3,12)
        p = dict(variables=variables,cap=cap,total=r.randint(cap+1,variables*cap-2))
    elif family == "coin_change":
        p = dict(coins=sorted([1]+r.sample(range(2,20),3)),amount=20+index%977)
    elif family == "subset_sum":
        values = [r.randint(-5,20) for _ in range(r.randint(8,16))]
        size = r.randint(3,6)
        p = dict(values=values,size=size,total=sum(r.sample(values,size)))
    elif family in {"derangements","involutions"}:
        p = dict(n=5+index%71)
    elif family == "surjections":
        p = dict(domain_size=10+index%83,codomain_size=r.randint(3,8))
    elif family == "catalan":
        p = dict(semilength=5+index%89)
    elif family == "necklaces":
        p = dict(length=3+index%89,colors=r.randint(2,6))
    elif family == "ballot":
        a = 5+index%89
        p = dict(a_votes=a,b_votes=r.randint(2,a-1))
    elif family == "permutation_inversions":
        n = 5+index%31
        p = dict(n=n,inversions=r.randint(2,n*(n-1)//2-2))
    elif family == "gcd_lcm":
        g = 2+index%997
        ratio = r.choice([6,12,30,42,60,70,90,210])
        p = dict(gcd=g,lcm=g*ratio)
    elif family == "divisor_sum":
        p = dict(n=60+index*19,power=r.randint(1,3))
    elif family == "totient":
        p = dict(n=100+index*37)
    elif family == "modular_power":
        p = dict(base=r.randint(3,100),exponent=50+index*13,modulus=r.choice([97,101,127,256,997]))
    elif family == "multiplicative_order":
        modulus = r.choice([97,101,127,257,503,997])
        p = dict(base=r.randint(2,modulus-1)+(index%7)*modulus,modulus=modulus)
    elif family == "valuation_factorial":
        p = dict(n=50+index*17,prime=r.choice([2,3,5,7,11,13]))
    elif family == "trailing_zeros_binomial":
        n = 30+index*11
        p = dict(n=n,k=r.randint(3,n-3))
    elif family == "diophantine_nonnegative":
        a,b = r.randint(3,25),r.randint(3,25)
        p = dict(a=a,b=b,total=a*r.randint(4,20)+b*r.randint(4,20)+index)
    elif family == "integer_triangles":
        p = dict(perimeter=30+index%251)
    elif family in {"triangle_area_squared","circumradius_squared"}:
        x,y = index%101,r.randint(-20,20)
        p = dict(points=[[x,y],[x+r.randint(3,20),y],[x+r.randint(-5,5),y+r.randint(3,20)]])
    elif family == "lattice_polygon_interior":
        x,y = index%101,r.randint(-20,20)
        w,h = r.randint(4,20),r.randint(4,20)
        p = dict(vertices=[[x,y],[x+w,y],[x+w,y+h-1],[x+w-1,y+h],[x,y+h]])
    elif family == "markov_chain":
        weights = [[r.randint(1,9) for _ in range(3)] for _ in range(3)]
        p = dict(transition=[[str(Fraction(v,sum(row))) for v in row] for row in weights],start=0,target=index%3,steps=r.randint(3,12))
    elif family == "urn_hypergeometric":
        red,blue = 3+index%97,r.randint(5,25)
        draws = r.randint(3,min(12,red+blue-1))
        p = dict(red=red,blue=blue,draws=draws,target_red=r.randint(max(0,draws-blue),min(red,draws)))
    elif family == "expected_max":
        p = dict(dice=r.randint(2,6),sides=2+index%97)
    else:
        raise ValueError("Unsupported parameter family")
    return {"family":family,"parameters":p}
