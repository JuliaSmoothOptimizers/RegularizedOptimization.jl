
function show_table(mp, vals, labs)
    df = DataFrame(hcat(["\$ f(x) \$", "\$ h(x) \$", "\$ \\|x - x_0\\|_2/\\|A\\| \$", "\$ \\nabla f \$ evals", "\$ \\prox{\\nu\\psi}\$ calls"], hcat([vals[:,k] for k in 1:length(labs)])), :auto)
    rename!(df,  vcat(:Function, [Symbol(labs[i]) for i in 1:length(labs)]), makeunique=true)
    return df
end


function write_table(mp, df, filename)

# Generate table header
  Table = "\\footnotesize\\setlength{\\tabcolsep}{3pt}\n"
  Table  *= "\\begin{tabular}{ cc " * "|ccc" ^ (length(mp)) * "}\n";
  Table *= "   &    & "
  for i = 1:length(mp)
    if i <length(mp)
      Table*="\\multicolumn{"*string(length(mp))*"}{|c|}{"*mp[i] * "} & "
    else
      Table*="\\multicolumn{"*string(length(mp))*"}{|c}{"*mp[i] * "} \\\\ \\hline \n "
    end
  end


  Table *= " "
  for i in 1:ncol(df) 
    if i==1
      Table *= " "
    else
      Table *= " & " * split(string(names(df)[i]), "_")[1]
    end
  end
  Table *= " \\\\\n";
  Table *= "    \\hline\n";

  Table *= "    % Table body\n";
  for row in 1 : nrow(df)
     Table *= "  "; 
    Table *= "  "; 
    for col in 1 : ncol(df)
      if col ==1 
        Table*= String(df[row,col])
      elseif (row==2 && col ==2)
        Table *= " & " * @sprintf("%d", df[row,col])*"/0 "
      # elseif (row==2 && (col>2 && col<6))
      #   Table *= " & " * @sprintf("%.3f", df[row,col])
      elseif (row==2 && col>6) || (row==3 && col==2) || (row > 3 && col > 2)
        Table *= " & " * @sprintf("%d", df[row,col])
      elseif ((row==4||row==5) && col==2)
        Table *= " &  "
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