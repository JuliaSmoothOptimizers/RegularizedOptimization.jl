using LinearAlgebra



function oneProjector(b,d,tau)
# % ----------------------------------------------------------------------

    # Initialization
   n = length(b);
   x = zeros(n,);

    # Check for quick exit.
   if (tau >= norm(d.*b,1))
      x = b; itn = 0; 
      return x, itn 
   end
   # if (tau <  eps)
   if (tau < 2.2204e-16)
      itn = 0; 
      return x, itn 
   end

    # Preprocessing (b is assumed to be >= 0)
   bd = b ./ d;
   idx = sortperm(bd, rev=true); # Descending.
   bd = bd[idx];
   b  = b[idx];
   d  = d[idx];
    # Optimize
   csdb = 0; csd2 = 0;
   soft = 0; alpha1 = 0; i = 1;
   while (i <= n)
      csdb = csdb + d[i].*b[i];
      csd2 = csd2 + d[i].*d[i];
  
      alpha1 = (csdb - tau) / csd2;
      alpha2 = bd[i];

      if alpha1 >= alpha2
         break;
      end
    
      soft = alpha1;  i = i + 1;
   end
   x[idx[1:i-1]] = b[1:i-1] - d[1:i-1] * max(0,soft);

    # Set number of iterations
   itn = i;
   return x, itn
end