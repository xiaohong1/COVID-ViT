function xg_create_CT_covid_mask()

%addpath NIfTI_tools;

inF = 'covid';
outF = 'covid-seg'
xg_mkdir(outF);
listD = dir([inF,'/ct*']); %list all 3D CT folders. %{'3D-covid'}
no = length(listD)
xCol = 440;
yRow = 360;
for kk = 76:no % 144,232=ct-306,236=ct-31,
    k = kk
    inFolder = listD(kk).name
    %inFolder = 'ct_scan_233'
    outF_seg = [outF,'/',inFolder];
    xg_mkdir(outF_seg);
    listF = dir([inF,'/',inFolder,'/*.jpg']);
    num = max(size(listF));
    
    im = imread([inF,'/',inFolder,'/','0.jpg']);
    [r0,c0,d0]=size(im);
    im1 = zeros(r0,c0,num);
    for jj = 1:  num
        name1 = [inF,'/',inFolder,'/',num2str(jj-1),'.jpg'] ;
        im = imread(name1);
        [r1,c1,d1] = size(im);
        if (r1~=r0) || (c1~=c0)
            continue;
        end
        im1(:,:,jj) = mat2gray(im(:,:));
    end  
    [row, col, dim] = size(im1);
    d_box= round(dim/2);
    if(dim < 2*d_box) 
        d_box=round((dim-10)/2); % find middle slice of 3D CT
    end
    cdd = round(dim/2);
    imNo =0;
    %{
    if dim >50
        cropped_1 = im1(41:500,171:340, cdd-15:cdd+15);
    else if dim>3
            cropped_1 = im1(41:500,171:340, cdd-1:cdd+1);
        else
          cropped_1 = im1(41:500,171:340, dim);  
        end
    end
    %}
    if dim >50
        cropped_1 = im1(:,:, cdd-15:cdd+15);
    else if dim>3
            cropped_1 = im1(:,:, cdd-1:cdd+1);
        else
          cropped_1 = im1(:,:, dim);  
        end
    end
    
   % figure(101), imshow(cropped_1(:,:,30));
        %  minI = min(min(min(I));
       %   maxI = max(max(max(I)));
     minI = min(min(min(cropped_1)));
     maxI = max(max(max(cropped_1)));
     meanI = mean(mean(mean(cropped_1)));
          
     [r,c,dd] = (find(im1<meanI)) ;   
      mean_rcdd = [min(r), rem(mean(c), col), mean(mean(c)/col)] ;   
      [d0, d1] = xg_bounding(round( mean(c)/col), d_box, dim);
      box_def=0;
      step = max(round((d1-d0)/100),1);
      for ii = d0:1:d1  %:cdd+ dd
          imNo = imNo+1;
          I = im1(:,:,ii);
      %    figure, imshow(I);
        %  cropped = I(1:510,21:480); %% Crop region of interest 
         % cropped = I(21:480,1:510);
          cropped = I(10:510,10:510);
          mx_cropped = max(max(cropped));
          thresholded = (cropped <(0.54)); %% Threshold to isolate lungs
         % figure(11),  
         % subplot(1,4,1), imshow(mat2gray(thresholded)), title('threholded');
 
          clearThresh =  imclearborder(thresholded); %% Remove border artifacts in image    
         % subplot(1,4,2), imshow(mat2gray(clearThresh)), title('thres-no-border');
          
        lung = bwareaopen(clearThresh,100); %% Remove objects less than 100 pixels
          % subplot(1,4,3), imshow(mat2gray(lung)), title('thres-no-small');
           lung1 = imfill(lung,'hole'); % fill in the vessels inside the lungs
          % subplot(1,4,4), imshow(mat2gray(lung1)), title('thres-no-hole');
           %{,
          se = strel('disk',30);
          lung1 =imdilate(lung1, se); % retain bourders
         % subplot(1,4,2), imshow(mat2gray(lung1)), title('thres-dilated');
          %}
          %  outname = [outFolder, '/t', num2str(kk), '-', name2,'-',num2str(imNo),'.jpg'];
             outname = [outF_seg, '/','covid_', inFolder, '-',num2str(ii), '.jpg'];
           outIm = mat2gray(lung1.*cropped);   
           % outIm1 = flip((imrotate(outIm,90)));
           % size(outIm1)
           [row, col,dim] = size(outIm);
           if box_def <1
               [yc,xc,z] = find(outIm>0.6);
               y0 = max(round(mean(yc))-(yRow/2)+30+1,1);
               y1 = yRow+y0-1;
               if y1 >(row-2)
                   y1 = row-2;
                   y0 = y1-yRow+1; %yRow=320
               end
               x0 = max(round(mean(xc))-(xCol/2)+1,1);
               x1 = xCol+x0-1;
               if x1 >(col-2)
                   x1 = col-2;
                   x0 = x1-xCol+1;
               end
               box_def =1;
           end
           x0_y1=[y0,y1,x0,x1, size(outIm)];
           
           outIm2= outIm(y0:y1,x0:x1); %(81:400,81:400);
          % outIm2 = flipud(outIm2);
           sz_Im2 = size(outIm2);
           if (mean(mean(outIm2)) > (60/255))
                imwrite(outIm2,outname);
                outname = ['test/','covid_', inFolder, '-',num2str(ii), '.jpg'];
                imwrite(outIm2,outname);
           end
           %se = strel('disk',30);
          % outIm3 =imdilate(outIm2, se);    %reparing broken edges
          outIm3 = mat2gray(lung1.*cropped);
         %  figure(3), imshowpair(mat2gray(outIm2),cropped,'montage')
           
          clear I outIm outIm1 outIm2 outIm3 
        end
        clear im Im1 im1
    end
end

%---------------------------------- bounding box----------%
%{,
function [b0, b1] = xg_bounding(centre, bounding, b_max)
    b0 = centre- bounding+1;
    b1 = centre + bounding; 
    
    if( b0 <=0) 
        b0 = 1;
        b1 = bounding *2;
    end
    if( b1 > b_max)
        b1 = b_max;
        b0 = b_max - bounding*2 +1;
    end
end
%}
%--------------------xg-mkdir--------%
function xg_mkdir(fld)
    if ~exist(fld,'dir')
        mkdir(fld);
    end
end
