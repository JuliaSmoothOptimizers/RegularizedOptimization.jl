
function show_table(mp, vals, labs)
    dp = DataFrame([mp[:,i] for i in 1:length(labs)])
    rename!(dp,  [Symbol(labs[i]) for i in 1:length(labs)])
    df = DataFrame(hcat(["\$ (f + h)(x) \$", "\$ f(x) \$", "\$ h(x) \$", "\$ \\frac{||x - x_0||_2}{||A||} \$", "\$ N \$"], hcat([vals[:,i] for i in 1:length(labs)])))
    rename!(df,  vcat(:Function, [Symbol(labs[i]) for i in 1:length(labs)]))
    return dp, df
end


function write_table(dp, df, filename)

# Generate table header
  Table  = "\\begin{tabular}{| " * "c |" ^ (ncol(dp)+1) * "}\n";
  Table *= "    \\hline\n";
  Table *= "  % Table header\n";
  # Table *= "    \\rowcolor[gray]{0.9}\n";
  Table *="\\multicolumn{"* string(ncol(df)) *"}{|c|}{Minima} \\\\ \\hline"
  Table *= " "
  for i in 1:ncol(df) 
    if i==1
      Table *= string(names(df)[i])
    else
      Table *= " & " * string(names(df)[i])
    end
  end
  # Table *= " " 
  # for i in 1:ncol(df)
  #     Table *= " & " * string(names(df)[i])
  # end
  Table *= " \\\\\n";
  Table *= "    \\hline\n";

# Generate table body (with nice alternating row colours)
  # toggleRowColour(x) = x == "0.8" ? "0.7" : "0.8";
  # rowcolour = toggleRowColour(0.7);

  Table *= "    % Table body\n";
  for row in 1 : nrow(df)
    # Table *= "  \\rowcolor[gray]{" * (rowcolour = toggleRowColour(rowcolour); rowcolour) * "}\n";
    Table *= "  "; 
    # for col in 1 : ncol(dp) 
    #   if col ==1
    #     Table *= @sprintf("%.3f", dp[row,col]);
    #   else
    #     Table *= " & " * @sprintf("%.3f", dp[row,col]); 
    #   end
    # end
    Table *= "  "; 
    for col in 1 : ncol(df)
      if col ==1 
        Table*= String(df[row,col])
      else
        Table *= " & " * @sprintf("%.3f", df[row,col])
      end 
    end
    Table *= " \\\\\n";
    Table *= "  \\hline\n"; 
  end
  Table *= "\\end{tabular}\n";

# Export result to .tex file
  write(string(filename,".tex"), Table);
  return Table
end