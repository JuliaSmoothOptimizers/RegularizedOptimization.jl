
function tab_preproc(f_obj, h_obj, xvars,pnumtab, hist, A, λ)

    ftab = zeros(length(xvars))
    htab = zeros(length(xvars))
    objtab = zeros(length(xvars))
    partab = zeros(length(xvars))
    numtab = zeros(length(xvars))
    for i = 1:length(xvars)
        ftab[i] = f_obj(xvars[i])[1] 
        htab[i] = h_obj(xvars[i])
        objtab[i] = ftab[i] + λ * htab[i]
        if isa(A, Array)
            partab[i] = norm(xvars[1] - xvars[i]) / opnorm(A)
        else
            partab[i] = norm(xvars[1] - xvars[i])
        end
        if i == 1
            numtab[i] = 0
        else
            numtab[i] = length(hist[i - 1])
        end
    end
    if isa(A, Array)
        objtest = abs(objtab[2] - objtab[1]) / opnorm(A)
    else 
        objtest = abs(objtab[2] - objtab[1])
    end
    
    partest = partab[2]



    # vals = vcat(objtab', ftab', htab', partab', numtab', pnumtab')
    vals = vcat(ftab', htab', partab', numtab', pnumtab')
    pars = hcat(xvars)

    return vals, pars
end