function d = dspd(t1, t2)
[ds1, dr1] = e_spd(t1,t2);
[ds2, dr2] = e_spd(t2,t1);

%d = min([ds1, ds2])*(1+min([dr1, dr2]));
d = min([ds1, ds2])*2 / (2 - min([dr1, dr2]) + 0.0000001);
%d = min([ds1, ds2])* exp(min([dr1, dr2]));
end % func


function [ds, dr] = e_spd(t1,t2)
if isempty(t1) || isempty(t2)
    ds = Inf;
    dr = Inf;
    return;
end

npts = size(t1,1);
ds = zeros(npts,1);
dr = zeros(npts,1);
direct = [t1(3,1:2)-t1(1,1:2); t1(3:end,1:2)-t1(1:end-2,1:2); t1(end,1:2)-t1(end-2,1:2)];
%direct = [t1(2:end,1:2)-t1(1:end-1,1:2); t1(end,1:2)-t1(end-1,1:2)];
for i=1:npts
    [ds(i), dr(i)] = point_to_trajectory(t1(i,:), direct(i,:), t2);
end
ds = mean(ds);
dr = mean(dr);
end % func



function [dpt, dpr] = point_to_trajectory(p,v, t)
if isempty(p) || size(t,1)<2
    dpt = Inf;
    dpr = Inf;
    return;
end
seg_nums = size(t,1) - 1;

dpt = Inf;
for i = 1:seg_nums
    tmp = point_to_seg(p, t(i,:), t(i+1,:) );
    if tmp < dpt
        dpt = tmp;
        dpr = i;
    end
end
if dpr < 2
    tmp = t(dpr+1,:)-t(dpr,:);
elseif dpr==length(t)-1
    tmp = t(dpr+1,:)-t(dpr,:);
else
    tmp = t(dpr+2,:)-t(dpr,:);
end 
dpr = 1 - sum(v.*tmp)/( sqrt( sum(v.^2)*sum(tmp.^2) ) );
end % 


function dpl = point_to_seg(p, s1, s2)
vec_AB = s2 - s1;
vec_AP = p - s1;
AP2 = sum(vec_AP.^2);
AB2 = sum(vec_AB.^2);

r = sum(vec_AB.*vec_AP)/AB2;

if r<=0
    dpl = sqrt( AP2 );
        
elseif r>=1
    dpl = sqrt( sum( (p - s2).^2 ) );
else
    dpl = sqrt( AP2 - AB2*(r^2) );
end
%}
end % func







