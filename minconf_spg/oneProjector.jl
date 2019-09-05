using LinearAlgebra



function oneProjector(b,d,tau)
# % ----------------------------------------------------------------------
#there is a non-scalar version of d too, but for now we just take the scalar value 
   if isa(d, Float64) && d==0
      x = b; 
      itn = 0.0;
      return x
   end

   #get signs of all elements in b
   s=sign.(b);
   b = abs.(b);
    # Initialization
   n = length(b);
   x = zeros(n,);

    # Check for quick exit.
   if (tau >= norm(b,1))
      x = b; itn = 0; 
      return x.*s 
   end
   # if (tau <  eps)
   if (tau < 2.2204e-16)
      itn = 0; 
      return x.*s 
   end

    # Preprocessing (b is assumed to be >= 0), taken care of with abs
   idx = sortperm(b, rev=true); # Descending.
   b  = b[idx];
    # Optimize
   csb = -tau; 
   alphaPrev = 0;  
   i = 1;
   while (i <= n)
      csb = csb + b[i];
  
      alpha = csb + b[i];

      if alpha >= b[i]
         break;
      end
    
      alphaPrev = alpha;  i = i + 1;
   end
   x[idx] = max.(0, b - alphaPrev*ones(size(b)))

    # Set number of iterations
   itn = i;
   return x.*s
end