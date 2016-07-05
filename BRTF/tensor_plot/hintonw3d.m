function ax = hintonw3d(w,max_m,min_m)
%HINTONW Hinton graph of weight matrix.
%
%  Syntax
%
%    hintonw(W,maxw,minw)
%
%  Description
%
%    HINTONW(W,MAXW,MINW) takes these inputs,
%      W    - SxR weight matrix
%      MAXW - Maximum weight, default = max(max(abs(W))).
%      MINW - Minimum weight, default = M1/100.
%    and displays a weight matrix represented as a grid of squares.
%
%    Each square's AREA represents a weight's magnitude.
%    Each square's COLOR represents a weight's sign.
%    RED for negative weights, GREEN for positive.
%
%  Examples
%
%    W = rands(4,5);
%    hintonw(W)
%
%  See also HINTONWB.

% Mark Beale, 1-31-92
% Revised 12-15-93, MB
% Revised 11-31-97, MB
% Copyright 1992-2007 The MathWorks, Inc.
% $Revision: 1.1.6.5 $  $Date: 2007/11/09 20:51:40 $

opengl software;

if nargin < 1,error('NNET:Arguments','Not enough input arguments.');end
if nargin < 2, max_m = max(max(max(abs(w)))); end
if nargin < 3, min_m = max_m / 100; end
if max_m == min_m, max_m = 1; min_m = 0; end

% DEFINE BOX EDGES
xn1 = [-1 -1 +1]*0.5;
xn2 = [+1 +1 -1]*0.5;
yn1 = [+1 -1 -1]*0.5;
yn2 = [-1 +1 +1]*0.5;

% DEFINE POSITIVE BOX
% xn = [-1 -1 +1 +1 -1]*0.5;
% yn = [-1 +1 +1 -1 -1]*0.5;

xn= [ -1   -1    -1   -1    -1   1;
    -1   -1    -1   -1    -1   1;
    1    1     1    1    -1   1;
    1    1     1    1    -1   1;
    -1   -1    -1   -1    -1   1;
    ] .*  0.5;
yn= [ -1    1    -1   -1     1   1;
    -1    1     1    1     1   1;
    -1    1     1    1    -1  -1;
    -1    1    -1   -1    -1  -1;
    -1    1    -1   -1     1   1;
    ] .*  0.5;
zn= [ -1   -1     1   -1    -1   1;
    1    1     1   -1     1  -1;
    1    1     1   -1     1  -1;
    -1   -1     1   -1    -1   1;
    -1   -1     1   -1    -1   1;
    ] .*  0.5;

verts =[-1 -1 -1;...
        -1 -1  1;...
        1  -1  1;...
        1  -1  -1;...
        -1  1  -1;...
        -1  1   1;...
        1   1   1;...
        1   1   -1
        ].*0.5;
faces =[ 1 2 3 4;...
         5 6 7 8;...
         2 6 7 3;...
         1 5 8 4;...
         1 2 6 5;...
         4 3 7 8
       ];    

% DEFINE POSITIVE BOX
% xp = [xn [-1 +1 +1 +0 +0]*0.5];
% yp = [yn [+0 +0 +1 +1 -1]*0.5];

[S,R,Q] = size(w);

% Added color difference
Colors = jet(256);

MaxV = max(w(:));
MinV = min(w(:));




cla reset;
hold on
set(gca,'xlim',[0.5-0.02 S+0.52]);
set(gca,'ylim',[0.5-0.02 R+0.52]);
set(gca,'zlim',[0.5-0.02 Q+0.52]);
set(gca,'xlimmode','manual');
set(gca,'ylimmode','manual');
set(gca,'zlimmode','manual');
xticks = get(gca,'xtick');
set(gca,'xtick',xticks(find(xticks == floor(xticks))))
yticks = get(gca,'ytick');
set(gca,'ytick',yticks(find(yticks == floor(yticks))))
zticks = get(gca,'ztick');
set(gca,'ztick',zticks(find(zticks == floor(zticks))))
set(gca,'ydir','reverse');
if get(0,'screendepth') > 1
%     set(gca,'color',[1 1 1].*1);
%       set(gca,'color','b');
end

for k=1:Q
    for i=1:S
        for j=1:R
            m = sqrt((abs(w(i,j,k))-min_m)/max_m);
            m = min(m,max_m)*1;
            Colidx = round((w(i,j,k)-MinV)./(MaxV-MinV).*(size(Colors,1)-1)+1);
            if isnan(Colidx)
               Colidx =  round(size(Colors,1)/2);
            end
            if real(m)
%                 if w(i,j) >= 0
%                     %       fill(xn*m+j,yn*m+i,[0 0.8 0])
%                     patch(xn*m+j, yn*m+i, zn*m+k,[0 0.8 0]);
%                     %         plot(xn1*m+j,yn1*m+i,'w',xn2*m+j,yn2*m+i,'k')
%                 elseif w(i,j) < 0
%                     %       fill(xn*m+j,yn*m+i,[0.8 0 0]);
%                     patch(xn*m+j, yn*m+i, zn*m+k,[0.8 0 0]);
%                     %         plot(xn1*m+j,yn1*m+i,'k',xn2*m+j,yn2*m+i,'w');
%                 end
%   patch by x, y, z coordinate 
%                 patch(xn*m+j, yn*m+i, zn*m+k,Colors(Colidx,:),'FaceLighting','phong');
%   patch by vertices and connection matrix
               p =  patch('Faces',faces,'Vertices',verts.*m + ones(size(verts,1),1)*[j, i, k],'FaceColor',Colors(Colidx,:), 'FaceLighting','phong');
               set(p, 'FaceAlpha', 0.2);               
            end
        end
    end
end
view(3);

% plot([0 R R 0 0]+0.5,[0 0 S S 0]+0.5,'w');
% verts = [0.5 0.5 Q+0.5;...
%     0.5 0.5 0.5;...
%     0.5 S+0.5 0.5;...
%     0.5 S+0.5 Q+0.5;...
%     R+0.5 0.5 Q+0.5;...
%     R+0.5 0.5 0.5;...
%     R+0.5 S+0.5 0.5;...
%     R+0.5 S+0.5 Q+0.5];
% faces =[1 2 3 4;...
%     2 3 7 6;...
%     4 3 7 8;...
%         1 2 6 5;
%         5 6 7 8;...
%     1 4 8 5;...
% ];
% patch('Faces',faces,'Vertices',verts,'FaceColor','none','LineWidth',0.5, 'EdgeColor','k');
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% grid on;
box on;
% camlight 

lighting phong
% alpha(0.8);
set(gca,'ZDir','reverse');

caxis([MinV MaxV]);
colorbar;
hold off;
ax = gca;
