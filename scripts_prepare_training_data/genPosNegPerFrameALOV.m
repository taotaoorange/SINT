%%% author: Ran Tao

nr_ang = 16;
r_step = 10;

IoU_thresh_p = 0.7;
IoU_thresh_n = 0.5;

%%
types = dir('../dataset/ALOV/imagedata++/'); types = types(3:end);

for i = 1:length(types)
   
    i
    videos = dir(['../dataset/ALOV/alov300++_rectangleAnnotation_full/' types(i).name '/*.ann']);
    
    for j = 1:length(videos)
        
       ann = load(['../dataset/ALOV/alov300++_rectangleAnnotation_full/' types(i).name '/' videos(j).name]);
       assert(size(ann,2) == 9);
       
       frameNames = dir(['../dataset/ALOV/imagedata++/' types(i).name '/' videos(j).name(1:end-4) '/*.jpg']);
       assert(isempty(frameNames)==0);
       
       num_samples = zeros(2, size(ann,1)); %%%%%
       
       for k = 1:size(ann,1) % number of annotated frames
          
           frameID = ann(k,1);
           im = imread(['../dataset/ALOV/imagedata++/' types(i).name '/' videos(j).name(1:end-4) '/' frameNames(frameID).name]);
           x = min(ann(k,2:2:end));
           y = min(ann(k,3:2:end));
           w = max(ann(k,2:2:end)) - min(ann(k,2:2:end)) + 1;
           h = max(ann(k,3:2:end)) - min(ann(k,3:2:end)) + 1;
          
           samples = funcSample(x,y,w,h,size(im,2),size(im,1),nr_ang,r_step);
           
           IoUs = zeros(1, size(samples,2));
           for m = 1:size(samples,2)
               IoUs(m) = funcIoU([samples(1,m),samples(2,m), samples(1,m)+samples(3,m)-1, samples(2,m)+samples(4,m)-1], [x,y,x+w-1,y+h-1]);
           end
           
           flags = zeros(1,size(samples,2));
           flags(IoUs>IoU_thresh_p) = 1;
           flags(IoUs<IoU_thresh_n) = -1;
           
           samples_pos = samples(:,flags==1); IoUs_pos = IoUs(flags==1);
           samples_neg = samples(:,flags==-1); IoUs_neg = IoUs(flags==-1);
           
           num_samples(1,k) = size(samples_pos,2) + 1; % +gt
           num_samples(2,k) = size(samples_neg,2);
           
           savefile = sprintf('%s_%s_%s.txt', types(i).name, videos(j).name(1:end-4), frameNames(frameID).name);
           fid = fopen(['PosNegPerFrameALOV/' savefile], 'w');
           fprintf(fid, '%d %f %f %f %f %f\n', 1, x, y, w, h, 1); % gt
           for pp = 1:size(samples_pos,2)
               fprintf(fid, '%d %f %f %f %f %f\n', 1, samples_pos(1,pp), samples_pos(2,pp), samples_pos(3,pp), samples_pos(4,pp), IoUs_pos(pp));
           end
           for nn = 1:size(samples_neg,2)
               fprintf(fid, '%d %f %f %f %f %f\n', 0, samples_neg(1,nn), samples_neg(2,nn), samples_neg(3,nn), samples_neg(4,nn), IoUs_neg(nn));
           end
        
           fclose(fid);
           
       end
        
       save(['PosNegPerFrameALOV_nr_ang16_r_step10_IoU_p0.7_n0.5/' videos(j).name(1:end-4) '.mat'], '-v7.3', 'num_samples');
       
        
    end
    
    
end