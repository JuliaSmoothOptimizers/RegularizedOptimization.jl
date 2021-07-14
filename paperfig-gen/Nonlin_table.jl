
function show_table(mp, vals, labs)
  dp = DataFrame([mp[:,i] for i in 1:length(labs)], :auto)
  rename!(dp,  [Symbol(labs[i]) for i in 1:length(labs)])
  df = DataFrame(hcat(["\$ f(x) \$", "\$ h(x) \$", "\$ ||x - x_0||_2 \$", "\$ \\nabla f \$ evals", "\$ \\prox{\\nu\\psi}\$ calls"], hcat([vals[:,i] for i in 1:length(labs)])), :auto)
  rename!(df,  vcat(:Function, [Symbol(labs[i]) for i in 1:length(labs)]))
  return dp, df
end


function write_table(dp, df, filename)

# Generate table header
  Table  = "\\footnotesize\n \\begin{tabular}{  " * "c  " ^ ncol(dp) * " ||  " * "c  " ^ ncol(df) * "}\n";
  Table *= "    % Table header\n";
  # Table *= "    \\rowcolor[gray]{0.9}\n";
  Table *="\\multicolumn{"* string(ncol(dp)) *"}{c||}{Parameters} & \\\\" #\\multicolumn{"*string(ncol(df))*"}{|c|}{Minima}\\\\ \\hline"
  Table *= " "
  for i in 1:ncol(dp) 
    if i==1
      Table *= string(names(dp)[i])
    else
      Table *= " & " * string(names(dp)[i])
    end
  end
  Table *= " " 
  for i in 1:ncol(df)
      Table *= " & " * string(names(df)[i])
  end
  Table *= " \\\\\n";
  Table *= "    \\hline\n";

  Table *= "    % Table body\n";
  for row in 1 : nrow(dp)
     Table *= "  "; 
    for col in 1 : ncol(dp) 
      if col ==1
        if dp[row, col]==0
          Table *= @sprintf("%d", dp[row,col]);
        else
          Table *= @sprintf("%.3f", dp[row,col]);
        end
      else
        if dp[row, col]==0
          Table *= " & " * @sprintf("%d", dp[row,col]); 
        else
          Table *= " & " * @sprintf("%.3f", dp[row,col]); 
        end
      end
    end
    Table *= "  "; 
    for col in 1 : ncol(df)
      if col ==1 
        Table*= " & " * String(df[row,col])
      elseif row==2 || (row==3 && col==2) || (row >3 && col>2)
        Table *= " & " * @sprintf("%d", df[row,col])
      elseif (row==4 || row==5)&& col ==2
        Table *= " & " * @sprintf(" ")
      else
        Table *= " & " * @sprintf("%.3f", df[row,col])
      end 
    end
    Table *= " \\\\\n";
  end
  Table *= "  \\hline\n"; 
  Table *= "\\end{tabular}\n";

# Export result to .tex file
  write(string(filename,".tex"), Table);
  return Table
end