export linesearch, lsTR, directsearch, directsearch!

function linesearch(x, zl, zu, s, dzl, dzu,l,u ;mult=.9, tau = .01)
	α = 1.0
	     while(
            any(x + α*s - l .< (1-tau)*(x-l)) ||
            any(u - x - α*s .< (1-tau)*(u-x)) ||
            any(zl + α*dzl .< (1-tau)*zl) ||
            any(zu + α*dzu .< (1-tau)*zu)
            )
            α = α*mult

        end
        return α
end

function lsTR(x, s,l,u ;mult=.9, tau = .01)
	α = 1.0
	     while(
            any(x + α*s - l .< (1-tau)*(x-l)) ||
            any(u - x - α*s .< (1-tau)*(u-x))
            )
            α = α*mult

        end
        return α
end

function directsearch(xsl, usx, zkl, zku, s, dzl, dzu; tau = .01) #used to be .01
	temp = [(-tau *(xsl))./s; (-tau*(usx))./-s; (-tau*zkl)./dzl; (-tau*zku)./dzu]
    temp=filter((a) -> 1>=a>0, temp)
    return minimum(vcat(temp, 1.0))
end

function ds(xsl, usx, s; tau = .01) #used to be .01
	temp = [(-tau *(xsl))./s; (-tau*(usx))./-s]
    temp=filter((a) -> 1>=a>0, temp)
    return minimum(vcat(temp, 1.0))
end

function directsearch!(xsl, usx,α, zkl, zku, s, dzl, dzu; tau = .01) #used to be .01
	temp = [(-tau *(xsl))./s; (-tau*(usx))./-s; (-tau*zkl)./dzl; (-tau*zku)./dzu]
    temp=filter((a) -> 1>=a>0, temp)
    α = minimum(vcat(temp, 1.0))
end