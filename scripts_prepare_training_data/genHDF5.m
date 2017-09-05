%%% author: Ran Tao

srcRoot = 'train_metafile_for_gen_hdf5_trial01'

num_pairs = 64631 %%%%%%%%%%

image_info_1 = cell(num_pairs,1);
image_info_2 = cell(num_pairs,1);


nr_box_perim = 64;
box_info = zeros(num_pairs, 5*nr_box_perim*2+2);
% 
% nr_box_perim = 64;


%%
image_pair_txt_files = dir([srcRoot '/*_image_pairs.txt']);

count = 0;
for i = 1:length(image_pair_txt_files)

    fid = fopen([srcRoot '/' image_pair_txt_files(i).name], 'r');
    A = textscan(fid, '%s %s\n');
    fclose(fid);
    
    if isempty(A{1})
        continue
    end
    
    image_info_1(count+1:count+length(A{1})) = strcat(image_pair_txt_files(i).name(1:end-15),A{1});
    image_info_2(count+1:count+length(A{2})) = strcat(image_pair_txt_files(i).name(1:end-15),A{2});
    
    
    binfo = load([srcRoot '/' image_pair_txt_files(i).name(1:end-15) 'box_pairs.txt']);
    assert(size(binfo,2) == size(box_info,2));
    assert(size(binfo,1) == length(A{1}));
    box_info(count+1:count+length(A{1}),:) = binfo;
   
    count = count + length(A{1});
    
    i

end
assert(count == num_pairs)

%%
% shuffle
rseed = 10; rng(rseed);
randp = randperm(num_pairs);

image_info_1 = image_info_1(randp);
image_info_2 = image_info_2(randp);
box_info = box_info(randp,:);

%
mean_v = load('ilsvrc_2012_mean.txt');
mean_v = reshape(mean_v, 256*256, 3);
mean_v = single(mean(mean_v,1))


% 
%block_size = 2000;
block_size = 200;

num_blocks = ceil(num_pairs / block_size)

imageSz = 512;
% imageSz = 224;
for i = 1:num_blocks
%     sprintf('hdf5_files/train_hdf5_part%03d_perblock%d_nr_ang16_r_step10_IoU_p0.7_n0.5_64boxesperim.h5', i, block_size)
    sprintf('hdf5_files/train_hdf5_imsz%d_part%03d_perblock%d_nr_ang16_r_step10_IoU_p0.7_n0.5_64boxesperim.h5', imageSz, i, block_size)

    tic;
    idx = (i-1)*block_size+1:min(i*block_size,num_pairs); num = numel(idx);
    
    image_info_1_part = image_info_1(idx);
    image_info_2_part = image_info_2(idx);
    box_info_part = box_info(idx,:);
    
    
    IM1 = zeros(num,imageSz,imageSz,3, 'single');
    IM2 = zeros(num,imageSz,imageSz,3, 'single');
    B1 = zeros(nr_box_perim*2*5,1,1,num, 'single');
    B2 = zeros(nr_box_perim*2*5,1,1,num, 'single');
    
    Labels = zeros(num, 1, 1, nr_box_perim * 2);
    
    
    for jj = 1:num
       
        %
        pos = strfind(image_info_1_part{jj}, '_');
        impath = sprintf('../dataset/ALOV/imagedata++/%s/%s_%s/%s', image_info_1_part{jj}(1:pos(1)-1), image_info_1_part{jj}(1:pos(1)-1), image_info_1_part{jj}(pos(1)+1:pos(2)-1), image_info_1_part{jj}(pos(2)+1:end));
        im = imread(impath); w = size(im,2); h = size(im,1);
        w_r_1 = imageSz / w; h_r_1 = imageSz / h;
        
        im = single( imresize(im, [imageSz imageSz]) );
        IM1(jj,:,:,1) = im(:,:,3) - mean_v(1); % B
        IM1(jj,:,:,2) = im(:,:,2) - mean_v(2); % G
        IM1(jj,:,:,3) = im(:,:,1) - mean_v(3); % R

        
        %
        pos = strfind(image_info_2_part{jj}, '_');
        impath = sprintf('../dataset/ALOV/imagedata++/%s/%s_%s/%s', image_info_2_part{jj}(1:pos(1)-1), image_info_2_part{jj}(1:pos(1)-1), image_info_2_part{jj}(pos(1)+1:pos(2)-1), image_info_2_part{jj}(pos(2)+1:end));
        im = imread(impath); w = size(im,2); h = size(im,1);
        w_r_2 = imageSz / w; h_r_2 = imageSz / h;
        
        im = single( imresize(im, [imageSz imageSz]) );
        IM2(jj,:,:,1) = im(:,:,3) - mean_v(1); % B
        IM2(jj,:,:,2) = im(:,:,2) - mean_v(2); % G
        IM2(jj,:,:,3) = im(:,:,1) - mean_v(3); % R

        %
        assert(box_info_part(jj,1) == nr_box_perim)
        assert(box_info_part(jj,2) == nr_box_perim)
        
        binfo = box_info_part(jj,3:end);
        binfo1 = binfo(1:end/2);
        binfo2 = binfo(end/2+1:end);
        
        ov_ratio1 = binfo1(5:5:end); assert(ov_ratio1(1)==1); % gt
        ov_ratio2 = binfo2(5:5:end); assert(ov_ratio2(1)==1); % gt
        
        boxes1 = reshape(binfo1, 5, nr_box_perim); boxes1 = single(boxes1(1:4,:));
        boxes2 = reshape(binfo2, 5, nr_box_perim); boxes2 = single(boxes2(1:4,:));
        
        % resize accordingly
        boxes1([1 3],:) =  boxes1([1 3],:) * w_r_1; boxes1([2 4], :) = boxes1([2 4], :) * h_r_1;
        boxes2([1 3],:) =  boxes2([1 3],:) * w_r_2; boxes2([2 4], :) = boxes2([2 4], :) * h_r_2;
        boxes1 = round(boxes1) - 1; % from 0 in caffe
        boxes2 = round(boxes2) - 1; % from 0 in caffe
        
        boxes1_ = [zeros(1,size(boxes1,2)); boxes1];
        boxes1_f = [repmat(boxes1_(:,1), 1, nr_box_perim) boxes1_];
        boxes2_ = [zeros(1,size(boxes2,2)); boxes2];
        boxes2_f = [boxes2_ repmat(boxes2_(:,1), 1, nr_box_perim)];
        
        B1(:,1,1,jj) = single(reshape(boxes1_f, size(boxes1_f,2)*5, 1));
        B2(:,1,1,jj) = single(reshape(boxes2_f, size(boxes2_f,2)*5, 1));
        
        %
        Labels(jj,:,:,:) = single([ov_ratio2>=0.5 ov_ratio1>=0.5]);
        
    end

    %% write to hdf5 file
    IM1 = permute(IM1, [3 2 4 1]); 
    IM2 = permute(IM2, [3 2 4 1]); 

    Labels = permute(Labels, [3 2 4 1]); Labels = single(Labels);

 
    %h5filename = sprintf('hdf5_files/train_hdf5_part%02d_nr_ang16_r_step10_IoU_p0.7_n0.5_64boxesperim.h5', i);
%     h5filename = sprintf('hdf5_files/train_hdf5_part%03d_perblock%d_nr_ang16_r_step10_IoU_p0.7_n0.5_64boxesperim.h5', i, block_size);
    h5filename = sprintf('hdf5_files/train_hdf5_imsz%d_part%03d_perblock%d_nr_ang16_r_step10_IoU_p0.7_n0.5_64boxesperim.h5', imageSz, i, block_size);

    h5create(h5filename, '/data', [imageSz, imageSz, 3, num], 'Datatype','single')
    h5create(h5filename, '/data_p', [imageSz, imageSz, 3, num], 'Datatype','single')
    h5create(h5filename, '/boxes', [nr_box_perim*2*5,1,1, num], 'Datatype','single')
    h5create(h5filename, '/boxes_p', [nr_box_perim*2*5,1,1, num], 'Datatype','single')
    h5create(h5filename, '/label', [1, 1, nr_box_perim*2, num], 'Datatype','single')
    
    h5write(h5filename, '/data', IM1)
    h5write(h5filename, '/data_p', IM2)
    h5write(h5filename, '/boxes', B1)
    h5write(h5filename, '/boxes_p', B2)
    h5write(h5filename, '/label', Labels)
    i
    toc;
end

















%%
