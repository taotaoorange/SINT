function samples = funcSample(x,y, w, h, im_w, im_h, nr_ang, r_step)

% x,y,w,h : topleft, width, height of gt box
% im_w, im_h: width, height of the image
% 
% author: Ran Tao

rad = max(w,h);
r_stepsize = rad / r_step;

cos_values = cos(0:2*pi/nr_ang:2*pi); cos_values = cos_values(1:end-1);
sin_values = sin(0:2*pi/nr_ang:2*pi); sin_values = -sin_values(1:end-1);

dxdys = zeros(2,r_step * nr_ang);

count = 0;
for ir = 1:r_step
    
   offset = r_stepsize * ir;
   for ia = 1:nr_ang
       
       dx = offset * cos_values(ia);
       dy = offset * sin_values(ia);
       
       count = count + 1;
       dxdys(:, count) = [dx;dy];
       
   end
    
    
end
dxdys(1,:) = dxdys(1,:) + x;
dxdys(2,:) = dxdys(2,:) + y;


flags = dxdys(1,:) > 0 & dxdys(2,:) > 0 & dxdys(1,:)+w<im_w & dxdys(2,:)+h<im_h;

samples = zeros(4,size(dxdys,2));
samples(1:2,:) = dxdys;
samples(3,:) = w;
samples(4,:) = h;
samples = samples(:,flags);




