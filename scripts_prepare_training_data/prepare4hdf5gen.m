%%% author: Ran Tao

types = dir('../dataset/ALOV/imagedata++/'); types = types(3:end-1); % long-duration videos are not used.

% videos also appear in OTB which we do not include in the training
common_videos = {'01-Light_video00007','01-Light_video00015', '01-Light_video00016', '02-SurfaceCover_video00012', '06-MotionSmoothness_video00019', '06-MotionSmoothness_video00016', ...
    '11-Occlusion_video00010', '08-Clutter_video00011', '11-Occlusion_video00024', '11-Occlusion_video00025', '09-Confusion_video00028', '01-Light_video00021'};

%%
destRoot = 'train_metafile_for_gen_hdf5_trial01'

num_boxes = 64; % 1 gt + 63 boxes per frame
rseed = 1; rng(rseed);

%%
count_image_pairs = 0; % train: 64631, val: 7684
% count_image_pairs_insuff_boxes = 0;

for i = 1:length(types)
   
    i  
    type_name = types(i).name
    
    videos = dir(['../dataset/ALOV/alov300++_rectangleAnnotation_full/' type_name '/*.ann']);
    
    for j = 1:length(videos)-1 %% leave the last video as val
% %     for j = length(videos):length(videos)
    
        video_name = videos(j).name(1:end-4);
        if ismember(video_name, common_videos) % overlap with tracking benchmark (do not consider it for training)
            continue
        end
        
        % alov ann file
        ann = load(['../dataset/ALOV/alov300++_rectangleAnnotation_full/' type_name '/' videos(j).name]); assert(size(ann,2) == 9);
        frameNames = dir(['../dataset/ALOV/imagedata++/' type_name '/' video_name '/*.jpg']); % all frames of the video
        
        fid_im = fopen([destRoot '/' video_name '_image_pairs.txt'],'w');
        fid_box = fopen([destRoot '/' video_name '_box_pairs.txt'],'w');
        
        ann = ann(1:2:end,:); %%%%%%%%%%%%%%%%%%%%%%%%%%%%subsampling

        for mm = 1:size(ann,1) % number of annotated frames
            %mm
            frameID1 = ann(mm,1);
            imname1 = frameNames(frameID1).name;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            box_anno1 = load(['PosNegPerFrameALOV_nr_ang16_r_step10_IoU_p0.7_n0.5/' type_name '_' video_name '_' imname1 '.txt']);
            %box_anno1 = load(['PosNegPerFrameALOV_initWH_ang10_step10_IoU_p0.5_n0.5_scales3/' type_name '_' video_name '_' imname1 '.txt']);
            assert(box_anno1(1,end)==1); % first is the gt box
           
            % turn to x1 y1 x2 y2
            box_anno1(:,4) = box_anno1(:,4) + box_anno1(:,2) - 1;
            box_anno1(:,5) = box_anno1(:,5) + box_anno1(:,3) - 1;
            
            for nn = mm+1:size(ann,1)
                %nn
                count_image_pairs = count_image_pairs + 1;
                
                frameID2 = ann(nn,1);
                imname2 = frameNames(frameID2).name;

              
                
                %% box
                box_anno2 = load(['PosNegPerFrameALOV_nr_ang16_r_step10_IoU_p0.7_n0.5/' type_name '_' video_name '_' imname2 '.txt']);
                %box_anno2 = load(['PosNegPerFrameALOV_initWH_ang10_step10_IoU_p0.5_n0.5_scales3/' type_name '_' video_name '_' imname2 '.txt']);
                assert(box_anno2(1,end)==1); % first is the gt box
                box_anno2(:,4) = box_anno2(:,4) + box_anno2(:,2) - 1;
                box_anno2(:,5) = box_anno2(:,5) + box_anno2(:,3) - 1;
                
                
                boxes1 = zeros(num_boxes,5); 
                boxes2 = zeros(num_boxes,5);
                boxes1(1,:) = box_anno1(1,2:end); % gt
                boxes2(1,:) = box_anno2(1,2:end); % gt
                
                box_anno1_ = box_anno1(2:end,:);
                box_anno2_ = box_anno2(2:end,:);
                
                randp1 = randperm(size(box_anno1_,1));
                boxes1(2:min(num_boxes-1,size(box_anno1_,1))+1,:) = box_anno1_(randp1(1:min(num_boxes-1,size(box_anno1_,1))),2:end);
                boxes1 = boxes1(1:min(num_boxes-1,size(box_anno1_,1))+1,:);
                
                randp2 = randperm(size(box_anno2_,1));
                boxes2(2:min(num_boxes-1,size(box_anno2_,1))+1,:) = box_anno2_(randp2(1:min(num_boxes-1,size(box_anno2_,1))),2:end);
                boxes2 = boxes2(1:min(num_boxes-1,size(box_anno2_,1))+1,:);
                
                % write to file
                if size(boxes1,1) < num_boxes || size(boxes2,1) < num_boxes
                    count_image_pairs = count_image_pairs - 1;
                    continue
                end
                
                boxes = [boxes1; boxes2];
                fprintf(fid_box, '%f %f ', size(boxes1,1), size(boxes2,1));
                for xx = 1:size(boxes,1)-1
                    fprintf(fid_box, '%f %f %f %f %f ', boxes(xx,:));
                end
                fprintf(fid_box, '%f %f %f %f %f\n', boxes(end,:));
                
                fprintf(fid_im, '%s %s\n', imname1, imname2); %%%% write to file

% % %                 if size(boxes1,1) < num_boxes || size(boxes2,1) < num_boxes
% % %                     count_image_pairs_insuff_boxes = count_image_pairs_insuff_boxes + 1;
% % %                 end

            end
            
        end
            
        fclose(fid_box);
        fclose(fid_im);
    end
    
end
        