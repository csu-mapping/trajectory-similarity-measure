function pdis = traj_dist_run(T, traj_dist_fnc, varargin)
n = length(T);

% pair-wise distance: [1,2],[1,3],...,[1,n],[2,3],...,[2,n],...,[n-1,n]
% squareform(pd)
pdis = zeros(1,n*(n-1)/2);
index = 0;
m = n*(n-1)/2;
h = waitbar(0,'Please wait...');

try
    for i=1:(n-1)
        for j=(i+1):n
            t1 = [(T(i).X(1:end-1))',(T(i).Y(1:end-1))'];
            t2 = [(T(j).X(1:end-1))',(T(j).Y(1:end-1))'];
            t1(isnan(t1(:,1))|isnan(t1(:,2)),:) = [];
            t2(isnan(t2(:,1))|isnan(t2(:,2)),:) = [];
            index = index + 1;
            d = traj_dist_fnc(t1, t2, varargin{:});
            pdis(index) = real(d);
            
            
            waitbar(index/m);
        end
    end
catch
    close(h);
end


end % func


