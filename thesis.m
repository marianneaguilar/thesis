%% Create matrices that will hold relationship between each day-variable
%% pair for each cluster
group0comp=zeros(1215,1215);
group1comp=zeros(1215,1215);
group2comp=zeros(1215,1215);
group3comp=zeros(1215,1215);
%% Create vector that will hold largest contributions to hidden layers
%% overall for each cluster
group0contr=zeros(1215,1);
group1contr=zeros(1215,1);
group2contr=zeros(1215,1);
group3contr=zeros(1215,1);
%% Find each iteration of the bootstrap data released from Python
for n=0:98
    %%Import the data
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4exampleidentityiteration',num2str(n),'.xlsx'),'Sheet1');
    raw = raw(2:end,2:end);
    %%Create output variable
    exampleidentity = reshape([raw{:}],size(raw));
    %%Clear temporary variables
    clearvars raw;
    %%Repeat for second cluster
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4example2identityiteration',num2str(n),'.xlsx'),'Sheet2');
    raw = raw(2:end,2:end);
    example2identity = reshape([raw{:}],size(raw));
    clearvars raw;
    %%Repeat for third cluster
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4example3identityiteration',num2str(n),'.xlsx'),'Sheet3');
    raw = raw(2:end,2:end);
    example3identity = reshape([raw{:}],size(raw));
    clearvars raw;
    %%Repeat for fourth cluster
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4example4identityiteration',num2str(n),'.xlsx'),'Sheet4');
    raw = raw(2:end,2:end);
    example4identity = reshape([raw{:}],size(raw));
    clearvars raw;
    %%For each day-variable pair, calculate the ratio of the dot product to
    %%the maximal possible dot product
    for i=0:1214
        v1=exampleidentity(i+1,:);
        v3=example2identity(i+1,:);
        v5=example3identity(i+1,:);
        v7=example4identity(i+1,:);
        for j=0:1214
            comp1=mod(i,27);
            comp2=mod(j,27);
            if comp1<=comp2
                v2=exampleidentity(j+1,:);
                v1abs=abs(v1);
                v2abs=abs(v2);
                v4=example2identity(j+1,:);
                v3abs=abs(v3);
                v4abs=abs(v4);
                v6=example3identity(j+1,:);
                v5abs=abs(v5);
                v6abs=abs(v6);
                v8=example4identity(j+1,:);
                v7abs=abs(v7);
                v8abs=abs(v8);
                group0comp(i+1,j+1)=group0comp(i+1,j+1)+dot(v1,v2)/dot(v1abs,v2abs);
                group1comp(i+1,j+1)=group1comp(i+1,j+1)+dot(v3,v4)/dot(v3abs,v4abs);
                group2comp(i+1,j+1)=group2comp(i+1,j+1)+dot(v5,v6)/dot(v5abs,v6abs);
                group3comp(i+1,j+1)=group3comp(i+1,j+1)+dot(v7,v8)/dot(v7abs,v8abs);
            end
        end
        %%Add dot products to find contribution
        group0contr(i+1)=group0contr(i+1)+dot(v1,v1);
        group1contr(i+1)=group1contr(i+1)+dot(v3,v3);
        group2contr(i+1)=group2contr(i+1)+dot(v5,v5);
        group3contr(i+1)=group3contr(i+1)+dot(v7,v7);
    end
end

%% Take average of students dot product calculation
group0comp=group0comp/99;
group1comp=group1comp/99;
group2comp=group2comp/99;
group3comp=group3comp/99;

%% Create matrix for averaged relationships for each variable for each cluster
group0vars=zeros(45,45);
group1vars=zeros(45,45);
group2vars=zeros(45,45);
group3vars=zeros(45,45);

%% Create matrix for negative averaged relationships for each variable for each cluster
group0high=zeros(45,45);
group1high=zeros(45,45);
group2high=zeros(45,45);
group3high=zeros(45,45);

%% Create matrix for positive averaged relationships for each variable for each cluster
group0pos=zeros(45,45);
group1pos=zeros(45,45);
group2pos=zeros(45,45);
group3pos=zeros(45,45);
for i=0:44
    for j=0:44
        begin=i*27+1;
        ed=(i+1)*27;
        begin2=j*27+1;
        ed2=(j+1)*27;
        det0=det(group0comp(begin:ed,begin2:ed2));
        det1=det(group1comp(begin:ed,begin2:ed2));
        det2=det(group2comp(begin:ed,begin2:ed2));
        det3=det(group3comp(begin:ed,begin2:ed2));
        group0vars(i+1,j+1)=det0;
        if det0<0
            group0high(i+1,j+1)=det0;
        else
            group0pos(i+1,j+1)=det0;
        end
        group1vars(i+1,j+1)=det1;
        if det1<0
            group1high(i+1,j+1)=det1;
        else
            group1pos(i+1,j+1)=det1;
        end
        group2vars(i+1,j+1)=det2;
        if det2<0
            group2high(i+1,j+1)=det2;
        else
            group2pos(i+1,j+1)=det2;
        end
        group3vars(i+1,j+1)=det3;
        if det3<0
            group3high(i+1,j+1)=det3;
        else
            group3pos(i+1,j+1)=det3;
        end
    end
end

%% Find maximum and minimal values for variables
temp0=reshape(group0vars,[2025,1]);
temp1=reshape(group1vars,[2025,1]);
temp2=reshape(group2vars,[2025,1]);
temp3=reshape(group3vars,[2025,1]);
top0=min(maxk(temp0,245));
top1=min(maxk(temp1,245));
top2=min(maxk(temp2,245));
top3=min(maxk(temp3,245));
bot0=max(mink(temp0,200));
bot1=max(mink(temp1,200));
bot2=max(mink(temp2,200));
bot3=max(mink(temp3,200));

%% Select for highest and lowest 200 values
for i=1:45
    for j=1:45
        if group0vars(i,j)<top0 & group0vars(i,j)>bot0
            group0vars(i,j)=0;
            group0high(i,j)=0;
            group0pos(i,j)=0;
        end
        if group1vars(i,j)<top1 & group1vars(i,j)>bot1
            group1vars(i,j)=0;
            group1high(i,j)=0;
            group1pos(i,j)=0;
        end
        if group2vars(i,j)<top2 & group2vars(i,j)>bot2
            group2vars(i,j)=0;
            group2high(i,j)=0;
            group2pos(i,j)=0;
        end
        if group3vars(i,j)<top3 & group3vars(i,j)>bot3
            group3vars(i,j)=0;
            group3high(i,j)=0;
            group3pos(i,j)=0;
        end
    end
end

%% Find maximum and minimal values for day-variables
temp0=reshape(group0comp,[1215*1215,1]);
temp1=reshape(group1comp,[1215*1215,1]);
temp2=reshape(group2comp,[1215*1215,1]);
temp3=reshape(group3comp,[1215*1215,1]);
top0=min(maxk(temp0,15915));
top1=min(maxk(temp1,15915));
top2=min(maxk(temp2,15915));
top3=min(maxk(temp3,15915));
bot0=max(mink(temp0,14700));
bot1=max(mink(temp1,14700));
bot2=max(mink(temp2,14700));
bot3=max(mink(temp3,14700));
group0comphigh=zeros(1215,1215);
group1comphigh=zeros(1215,1215);
group2comphigh=zeros(1215,1215);
group3comphigh=zeros(1215,1215);

%% Select for highest and lowest values
for i=1:1215
    for j=1:1215
        if group0comp(i,j)>=top0 || group0comp(i,j)<=bot0
            group0comphigh(i,j)=group0comp(i,j);
        end
        if group1comp(i,j)>=top1 || group1comp(i,j)<=bot1
            group1comphigh(i,j)=group1comp(i,j);
        end
        if group2comp(i,j)>=top2 || group2comp(i,j)<=bot2
            group2comphigh(i,j)=group2comp(i,j);
        end
        if group3comp(i,j)>=top3 || group3comp(i,j)<=bot3
            group3comphigh(i,j)=group3comp(i,j);
        end
    end
end

%% Import the data for variable names
[~, ~, ofinterestnames] = xlsread('/Users/marianneaguilar/Documents/ofinterestnames.xlsx','Sheet1');
ofinterestnames = ofinterestnames(2:end,end);
ofinterestnames(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),ofinterestnames)) = {''};
dayvarpair=cell(1214,1);

%% Improve string to remove spaces or dashes
for i=1:length(ofinterestnames)
    ofinterestnames(i)=strrep(ofinterestnames(i),'_',' ');
    for j=1:27
        dayvarpair((i-1)*27+j)=strcat(ofinterestnames(i),' on day ',num2str(j));
    end
end

