function xg_2DTo3D()

dir_3d = 'train-3d-32/';
xg_create_dir(dir_3d);
step = 32;
overlap=0;
wh=224;
% total train subjects: covid 685(val:164), non-vovid 864 (val:179)
 for i =1: 865
    ii = i
    fnm0 = ['noncovid_ct_scan_',num2str(i-1)]
    fnm1 = dir(['train-gray/',fnm0,'-*.jpg']);
    no = length(fnm1);
    no_3d = floor(no/step);
    if no_3d <1
        continue;
    end
    %figure(1),
    for j=1:1:no_3d
        img_3d = zeros(wh,wh,step);
        idx = 1;
        for k=j:no_3d:(no_3d*step)   % each 3d Image set covers slices evenly distributed
            %idx = (j-1) *(step-overlap) +k;
            if idx>no
                continue;
            end
           % name = ['train-gray/',fnm1(idx).name];
            name = ['train-gray/',fnm1(k).name]
            im0 = imread(name);
            img1 = imresize(im0,[wh,wh]);
           % subplot(1,4,k), imshow(img1);
            img_3d(:,:,idx) = imrotate(img1,-90);
            %img_3d(:,:,idx) = img1;
            idx = idx+1;
        end
        %subplot(1,4,4), imshow(uint8(img_3d));
        out_fnm = [dir_3d,fnm0,'-',num2str(j)];
       % imwrite(uint8(img_3d),[out_fnm,'.jpg']);
        writeanalyze(img_3d,out_fnm,size(img_3d),'uint8');
        clear im0 img1 img_3d
        %{
        % displat to check
         info = analyze75info(out_fnm);
        im3 = analyze75read(info);
        imshow(im3(:,:,10));
        %}
    end
   
end
end

function xg_create_dir(dir0)
    if ~exist(dir0,'dir')
        mkdir(dir0);
    end
end