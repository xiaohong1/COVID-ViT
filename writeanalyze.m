function [] = writeanalyze(data,varargin)
% WRITEANALYZE - Save MRI data in the Anaylze format
%
%     usage:
%            writeanalyze(data,filename,vsize)
%            writeanalyze(data,filename,vsize,dtype)
%            writeanalyze(data,isize,filename,vsize)
%            writeanalyze(data,isize,filename,vsize,index)
%            writeanalyze(data,isize,filename,vsize,dtype)
%            writeanalyze(data,isize,filename,vsize,dtype,index)
%            writeanalyze('header',isize,filename,vsize,dtype)
%
%       data     - mri data
%                  Data can be a matrix(3d,4d) or vector. If data is a 
%                  vector then isize must be included. The data can also
%                  be indexed, meaning that each point corresponds to an
%                  indexed point in the 3d volume. In this case both the
%                  index vector and isize must be included.
%                  ( Note: 3d_data(index) = indexed_data 
%                    for 4d data the same index is used for each time
%                    point )
%       isize    - size of data [X,Y,Z] or [X,Y,Z,T]
%       filename - name of output file (with or without extension)
%       vsize    - size of voxels [X,Y,Z] (mm)
%       dtype    - data type ('uint8','int16','double',etc...)
%                  The default data type is the same as the input data
%                  matrix. Use this parameter to set the data type to be
%                  something different.
%       index    - index vector.
%
%       If the 'header' option is used writeanalyze will only write a
%       header file. This is useful if the data file already exists and
%       a header file is needed.
%

% Written by Colin Humphries
% Feb, 2000
% University of California, Irvine
% colin@alumni.caltech.edu

% Updated Sep 2003 - Changed parameter list
%                  - Can now handle 4-d data
%         Feb 2004 - Added SPM parameters

% Copywrite (c) 2003, Colin J. Humphries, All Rights Reserved

% Info on Analyze format: 
%    http://www.mayo.edu/bir/PDF/ANALYZE75.pdf
%

DEFAULT_SCALING = 'pos';  % Note: this variable is included to work with
			  % AIR which uses the global scaling variables
			  % which can be set in the header. Most programs
			  % don't use these variables.
			  % Values: 
			  % 'pos' 0 and largest possible value. ie for
			  %    uint8 [0 255] or for 'int16' [0 32767]
			  % 'posneg' smallest and largest possible values
			  %    ie for int16 [-32767 32767]
			  % 'auto' checks if there are any negative
                          %    values in which case 'posneg' is used
                          %    otherwise 'pos' is used.
			  % [] use the min and max of the data
			  % If you don't use this variable then you might
                          % as well use [0,0] which will set the bits to
                          % zeros.
			  
if ~isstr(data)
  isize = [];
  dtype = [];
  index = [];
  % Get arguments from list
  switch nargin
    case 6
      isize = varargin{1};
      filename = varargin{2};
      vsize = varargin{3};
      dtype = varargin{4};
      index = varargin{5};
    case 5
      isize = varargin{1};
      filename = varargin{2};
      vsize = varargin{3};
      if isstr(varargin{4})
	dtype = varargin{4};
      else
	index = varargin{4};
      end
    case 4
      if isstr(varargin{1})
	filename = varargin{1};
	vsize = varargin{2};
	dtype = varargin{3};
      else
	isize = varargin{1};
	filename = varargin{2};
	vsize = varargin{3};
      end
    case 3
      filename = varargin{1};
      vsize = varargin{2};
    otherwise
      error('Wrong number of inputs.');
  end

  % Get isize and dtype from the data
  if isempty(isize)
    isize = size(data);
  end  
  if isempty(dtype)
    dtype = class(data);
  end
  
  % Get rid of possible file extensions in filename
  ind = findstr(filename,'.img');
  if ~isempty(ind)
    filename = filename(1:ind-1);
  end
  ind = findstr(filename,'.hdr');
  if ~isempty(ind)
    filename = filename(1:ind-1);
  end

  % If the type is logical then use uint8
  if strcmp(dtype,'logical')
    dtype = 'uint8';
  end

  if length(isize) == 3
    % Output 3-d data
    if ~isempty(index)
      ndata = zeros(1,prod(isize));
      ndata(index) = data;
    else
      ndata = data;
    end
    if isempty(DEFAULT_SCALING)
      DEFAULT_SCALING = [min(ndata(:)) max(ndata(:))];
    end
    if isstr(DEFAULT_SCALING)
      if strcmp(DEFAULT_SCALING,'auto')
	if any(ndata(:) < 0)
	  DEFAULT_SCALING = 'posneg';
	else
	  DEFAULT_SCALING = 'pos';
	end
      end
    end
    
    % Write header file
    writeheaderfile(filename,isize,vsize,dtype,DEFAULT_SCALING);
    
    % Open img file
    fid = fopen([filename,'.img'],'w');
    if fid < 0
      error('Can''t open .img file.');
    end
    % Output data
    fwrite(fid,ndata,dtype);
    fclose(fid);
  else
    % Output 4-d data 
    % The 4d data is treated separately here because the indexing is
    % different.
    if isempty(DEFAULT_SCALING)
      DEFAULT_SCALING = [min(data(:)) max(data(:))];
    end
    if isstr(DEFAULT_SCALING)
      if strcmp(DEFAULT_SCALING,'auto')
	if any(ndata(:) < 0)
	  DEFAULT_SCALING = 'posneg';
	else
	  DEFAULT_SCALING = 'pos';
	end
      end
    end
    
    % Write header file
    writeheaderfile(filename,isize,vsize,dtype,DEFAULT_SCALING);
    
    % Open img file
    fid = fopen([filename,'.img'],'w');
    if fid < 0
      error('Can''t open .img file.');
    end
    if ~isempty(index)
      for ii = 1:isize(4)
	ndata = zeros(1,prod(isize(1:3)));
	ndata(index) = data((ii-1)*length(index)+1:ii*length(index));
	fwrite(fid,ndata,dtype);
      end
    else
      fwrite(fid,data,dtype);
    end
    
    fclose(fid);
    
  end

else
  if strcmp(data,'header')
    % Here we output only a header file.
    if nargin ~= 5
      error('Wrong number of input arguments')
    end
    isize = varargin{1};
    filename = varargin{2};
    vsize = varargin{3};
    dtype = varargin{4};
    
    % Get rid of possible file extensions
    ind = findstr(filename,'.img');
    if ~isempty(ind)
      filename = filename(1:ind-1);
    end
    ind = findstr(filename,'.hdr');
    if ~isempty(ind)
      filename = filename(1:ind-1);
    end
    if strcmp(DEFAULT_SCALING,'auto')
      DEFAULT_SCALING = 'posneg';
    end
    writeheaderfile(filename,isize,vsize,dtype,DEFAULT_SCALING);
  end  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function [] = writeheaderfile(filename,isize,vsize,dtype,scaling)
% Internal function for creating a header file.
%

% Switch from matlab datatype to the AIR datatype
switch dtype
  case 'uint8'
    datatype = 2;  
    bitpix = 8;
  case {'int16','short'}
    datatype = 4;
    bitpix = 16; 
  case {'int32','int'}
    datatype = 8;
    bitpix = 32;
  case {'single','float32','float'}
    datatype = 16;
    bitpix = 32;
  case {'double','float64'}
    datatype = 64;
    bitpix = 64;
  otherwise
    error('Unsupported datatype.');
end

% Set the scaling values.
if isstr(scaling)
  switch scaling
    case {'default','pos'}
      if bitpix == 8
	scaling = [0 255];
      elseif bitpix == 16
	scaling = [0 32767];
      else
	% Note: currently only 8 and 16 bit integers are set correctly.
	% because AIR only uses those datatypes.
	scaling = [0 32767];
      end
    case 'posneg'
      if bitpix == 8
	scaling = [0 255];
      elseif bitpix  == 16
	scaling = [-32767 32767];
      else
	% Note: currently only 8 and 16 bit integers are set correctly.
	scaling = [-32767 32767];
      end
  end
end
% Open header file	      
fid = fopen([filename,'.hdr'],'w');
if fid < 0
  error('Can''t open header file');
end

fwrite(fid,zeros(1,348),'uint8');

fseek(fid,0,'bof');
fwrite(fid,348,'int16');

fseek(fid,32,'bof');
fwrite(fid,16384,'int16');

fseek(fid,38,'bof');
fwrite(fid,'r','char');

fseek(fid,40+0,'bof');
% This is the convention used for isize that I've seen in other Analyze
% files. You could also use this convention (eg for 3d data
% fwrite(fid,[3,isize(1),isize(2),isize(3)],'int16'); ) and it should
% still be valid.
switch length(isize)
  case 1
    fwrite(fid,[4,isize(1),1,1,1],'int16');
  case 2
    fwrite(fid,[4,isize(1),isize(2),1,1],'int16');
  case 3
    fwrite(fid,[4,isize(1),isize(2),isize(3),1],'int16');
  case 4
    fwrite(fid,[4,isize(1),isize(2),isize(3),isize(4)],'int16');
end

fseek(fid,40+30,'bof');
fwrite(fid,datatype,'int16');

fseek(fid,40+32,'bof');
fwrite(fid,bitpix,'int16');

fseek(fid,40+36,'bof');
fwrite(fid,[0,vsize(:)'],'float32');

fseek(fid,40+100,'bof');
fwrite(fid,[scaling(2) scaling(1)],'int32');

% Note the following two fields are used by SPM and are not part of the
% Analyze format.
% Scale factor = 1
fseek(fid,40+72,'bof');
fwrite(fid,1,'int32');
% DC offset = 0
fseek(fid,40+76,'bof');
fwrite(fid,0,'int32');

fclose(fid);



