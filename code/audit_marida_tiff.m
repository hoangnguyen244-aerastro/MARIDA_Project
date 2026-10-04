%% audit_marida_tiff.m
% Audit MARIDA GeoTIFF structure before defining spectral representations.
% Does NOT assume that MATLAB channel indices correspond to Sentinel-2 bands.
% Compatible with MATLAB R2015b.

function audit_marida_tiff()
    fprintf('\n========================================\n');
    fprintf('MARIDA TIFF STRUCTURE AUDIT\n');
    fprintf('========================================\n');

    root='../raw_data/MARIDA/patches';
    D=dir(root);
    dirs=D([D.isdir]);
    dirs=dirs(~ismember({dirs.name},{'.','..'}));
    if isempty(dirs), error('No patch directories found under %s',root); end

    % Deterministic coverage across scene directories.
    max_files=20;
    paths={};
    for d=1:numel(dirs)
        F=dir(fullfile(root,dirs(d).name,'*.tif'));
        F=F(cellfun(@isempty,strfind({F.name},'_cl.tif')));
        F=F(cellfun(@isempty,strfind({F.name},'_conf.tif')));
        if ~isempty(F)
            paths{end+1}=fullfile(root,dirs(d).name,F(1).name); %#ok<AGROW>
        end
        if numel(paths)>=max_files, break; end
    end
    if isempty(paths), error('No image TIFF files found.'); end

    fid=fopen('../results/tiff_audit.txt','w');
    if fid<0, error('Cannot create ../results/tiff_audit.txt'); end;
    cleanup=onCleanup(@() fclose(fid));

    emit(fid,'MARIDA TIFF STRUCTURE AUDIT\n');
    emit(fid,'Files inspected: %d\n\n',numel(paths));

    channel_counts=zeros(numel(paths),1);
    for i=1:numel(paths)
        p=paths{i};
        info=imfinfo(p);
        I=imread(p);
        sz=size(I);
        if ndims(I)<3, nch=1; else nch=sz(3); end
        channel_counts(i)=nch;

        emit(fid,'[%02d] %s\n',i,p);
        emit(fid,'  imread size: %s | class: %s | ndims: %d | channels: %d\n', ...
            mat2str(sz),class(I),ndims(I),nch);
        emit(fid,'  imfinfo entries/pages: %d\n',numel(info));
        emit(fid,'  first page: Width=%d Height=%d BitDepth=%d SamplesPerPixel=%d\n', ...
            info(1).Width,info(1).Height,info(1).BitDepth,info(1).SamplesPerPixel);

        X=double(I);
        for b=1:nch
            if nch==1, v=X(:); else tmp=X(:,:,b); v=tmp(:); end
            v=v(isfinite(v));
            if isempty(v)
                emit(fid,'    channel %02d: no finite values\n',b);
            else
                emit(fid,'    channel %02d: min=%g max=%g mean=%.6g std=%.6g zero=%.3f%%\n', ...
                    b,min(v),max(v),mean(v),std(v),100*sum(v==0)/numel(v));
            end
        end
        emit(fid,'\n');
    end

    emit(fid,'SUMMARY\n');
    emit(fid,'Unique channel counts returned by imread: %s\n',mat2str(unique(channel_counts)'));
    emit(fid,['IMPORTANT: This audit intentionally does not assign channel numbers to B01/B02/... .\n' ...
        'Band mapping must be confirmed from MARIDA metadata/source before spectral ablation.\n']);

    fprintf('\nSaved ../results/tiff_audit.txt\n');
    fprintf('Commit that file after running this audit; do not rerun the full pipeline yet.\n');
end

function emit(fid,varargin)
    fprintf(varargin{:});
    fprintf(fid,varargin{:});
end
