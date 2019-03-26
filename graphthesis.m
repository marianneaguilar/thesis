%% Create graph for cluster 1 of all variable relationships, negative and positive relationships
d0=graph(group0vars,ofinterestnames,'omitselfloops');
h0=graph(group0high,ofinterestnames,'omitselfloops');
j0=graph(group0pos,ofinterestnames,'omitselfloops');
p0=plot(d0,'Layout','force','UseGravity','on')
highlight(p0,h0,'EdgeColor','r')
p=p0.NodeLabel;
p0.NodeLabel='';
text(get(p0,'XData'),get(p0,'YData'),p,'FontSize',12,'FontWeight','bold','HorizontalAlignment','left','VerticalAlignment','middle')
cla;
%% Create graph for cluster 2 of all variable relationships, negative and positive relationships
d1=graph(group1vars,ofinterestnames,'omitselfloops');
h1=graph(group1high,ofinterestnames,'omitselfloops');
j1=graph(group1pos,ofinterestnames,'omitselfloops');
p1=plot(d1,'Layout','force','UseGravity','on')
highlight(p1,h1,'EdgeColor','r')
p=p1.NodeLabel;
p1.NodeLabel='';
text(get(p1,'XData'),get(p1,'YData'),p,'FontSize',12,'FontWeight','bold','HorizontalAlignment','left','VerticalAlignment','middle')
cla;
%% Create graph for cluster 3 of all variable relationships, negative and positive relationships
d2=graph(group2vars,ofinterestnames,'omitselfloops');
h2=graph(group2high,ofinterestnames,'omitselfloops');
j2=graph(group2pos,ofinterestnames,'omitselfloops');
p2=plot(d2,'Layout','force','UseGravity','on')
highlight(p2,h2,'EdgeColor','r')
p=p2.NodeLabel;
p2.NodeLabel='';
text(get(p2,'XData'),get(p2,'YData'),p,'FontSize',12,'FontWeight','bold','HorizontalAlignment','left','VerticalAlignment','middle')
cla;
%% Create graph for cluster 4 of all variable relationships, negative and positive relationships
d3=graph(group3vars,ofinterestnames,'omitselfloops');
h3=graph(group3high,ofinterestnames,'omitselfloops');
j3=graph(group3pos,ofinterestnames,'omitselfloops');
p3=plot(d3,'Layout','force','UseGravity','on')
highlight(p3,h3,'EdgeColor','r')
p=p3.NodeLabel;
p3.NodeLabel='';
text(get(p3,'XData'),get(p3,'YData'),p,'FontSize',12,'FontWeight','bold','HorizontalAlignment','left','VerticalAlignment','middle')
cla;

%% Create directed graph (not for display) of day-variable relationships for cluster 1
tempgraph=digraph(group0comphigh,dayvarpair,'omitselfloops');
%% Find degree and centrality of each day-variable
degr=outdegree(tempgraph);
cen0=centrality(tempgraph,'betweenness');
%% Repeat for cluster 2
tempgraph2=digraph(group1comphigh,dayvarpair,'omitselfloops');
degr2=outdegree(tempgraph2);
cen1=centrality(tempgraph2,'betweenness');
%% Repeat for cluster 3
tempgraph3=digraph(group2comphigh,dayvarpair,'omitselfloops');
degr3=outdegree(tempgraph3);
cen2=centrality(tempgraph3,'betweenness');
%% Repeat for cluster 4
tempgraph4=digraph(group3comphigh,dayvarpair,'omitselfloops');
degr4=outdegree(tempgraph4);
cen3=centrality(tempgraph4,'betweenness');

%% Find average measure of centrality
avg0=(degr+cen0)/2;
avg1=(degr2+cen1)/2;
avg2=(degr3+cen2)/2;
avg3=(degr4+cen3)/2;

%% Nodes with maximal centrality measures (i.e. central hubs)
max0=[200 201 204 208 817];
max1=[201 416 420 422 817];
max2=[201 415 416 661 817];
max3=[201 202 661 686 817];

%% Matrix to find shortest paths
group0compfinal=zeros(1215,5);
group1compfinal=zeros(1215,5);
group2compfinal=zeros(1215,5);
group3compfinal=zeros(1215,5);

%% Find shortest path from each central hub to every other node and calculate projected effect for 0.9 retention
for i=1:length(max0)
    r1=max0(i);
    r2=max1(i);
    r3=max2(i);
    r4=max3(i);
    for j=1:1215
        v=shortestpath(tempgraph,r1,j,'Method','unweighted');
        s=1;
        for k=2:length(v)
            s=s*abs(group0comphigh(v(k-1),v(k)))/group0comphigh(v(k-1),v(k));
        end
        t=s*(0.9)^(length(v)-1);
        if abs(t)<1
            group0compfinal(j,i)=t;
        end
        v1=shortestpath(tempgraph2,r2,j,'Method','unweighted');
        s=1;
        for k=2:length(v1)
            s=s*abs(group1comphigh(v1(k-1),v1(k)))/group1comphigh(v1(k-1),v1(k));
        end
        t=s*(0.9)^(length(v1)-1);
        if abs(t)<1
            group1compfinal(j,i)=t;
        end
        v2=shortestpath(tempgraph3,r3,j,'Method','unweighted');
        s=1;
        for k=2:length(v2)
            s=s*abs(group2comphigh(v2(k-1),v2(k)))/group2comphigh(v2(k-1),v2(k));
        end
        t=s*(0.9)^(length(v2)-1);
        if abs(t)<1
            group2compfinal(j,i)=t;
        end
        v3=shortestpath(tempgraph4,r4,j,'Method','unweighted');
        s=1;
        for k=2:length(v3)
            s=s*abs(group3comphigh(v3(k-1),v3(k)))/group3comphigh(v3(k-1),v3(k));
        end
        t=s*(0.9)^(length(v3)-1);
        if abs(t)<1
            group3compfinal(j,i)=t;
        end
    end
end

%% Import the weight matrices from Python for cluster 1
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4exampleidentity.xlsx'),'Sheet1');
    raw = raw(2:end,2:end);
    %% Create output variable
    exampleidentity = reshape([raw{:}],size(raw));
    %% Clear temporary variables
    clearvars raw;
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4example2identity.xlsx'),'Sheet2');
    raw = raw(2:end,2:end);
    example2identity = reshape([raw{:}],size(raw));
    clearvars raw;
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4example3identity.xlsx'),'Sheet3');
    raw = raw(2:end,2:end);
    example3identity = reshape([raw{:}],size(raw));
    clearvars raw;
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4example4identity.xlsx'),'Sheet4');
    raw = raw(2:end,2:end);
    example4identity = reshape([raw{:}],size(raw));
    clearvars raw;
    
%% Import the weight matrices from Python for cluster 2
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4exampleidentity2ndlayer.xlsx'),'Sheet1');
    raw = raw(2:end,2:end);
    %% Create output variable
    exampleidentity2ndlayer = reshape([raw{:}],size(raw));
    %% Clear temporary variables
    clearvars raw;
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4example2identity2ndlayer.xlsx'),'Sheet2');
    raw = raw(2:end,2:end);
    example2identity2ndlayer = reshape([raw{:}],size(raw));
    clearvars raw;
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4example3identity2ndlayer.xlsx'),'Sheet3');
    raw = raw(2:end,2:end);
    example3identity2ndlayer = reshape([raw{:}],size(raw));
    clearvars raw;
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4example4identity2ndlayer.xlsx'),'Sheet4');
    raw = raw(2:end,2:end);
    example4identity2ndlayer = reshape([raw{:}],size(raw));
    clearvars raw;
    
%% Import the weight matrices from Python for cluster 3
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4exampleidentity3rdlayer.xlsx'),'Sheet1');
    raw = raw(2:end,2:end);
    %% Create output variable
    exampleidentity3rdlayer = reshape([raw{:}],size(raw));
    %% Clear temporary variables
    clearvars raw;
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4example2identity3rdlayer.xlsx'),'Sheet2');
    raw = raw(2:end,2:end);
    example2identity3rdlayer = reshape([raw{:}],size(raw));
    clearvars raw;
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4example3identity3rdlayer.xlsx'),'Sheet3');
    raw = raw(2:end,2:end);
    example3identity3rdlayer = reshape([raw{:}],size(raw));
    clearvars raw;
    [~, ~, raw] = xlsread(strcat('/Users/marianneaguilar/Documents/4example4identity3rdlayer.xlsx'),'Sheet4');
    raw = raw(2:end,2:end);
    example4identity3rdlayer = reshape([raw{:}],size(raw));
    clearvars raw;
    
%% Calculate net effect of one standard deviation change on three mood measures using weight matrices and distances    
neteffect=zeros(5,3);
neteffect2=zeros(5,3);
neteffect3=zeros(5,3);
neteffect4=zeros(5,3);
for k=1:5
    for j=1:3
        neteffect(k,j)=sum(transpose(exampleidentity3rdlayer(:,j))*(transpose(exampleidentity2ndlayer)*(transpose(exampleidentity)*group0compfinal(:,k))));
        neteffect2(k,j)=sum(transpose(example2identity3rdlayer(:,j))*(transpose(example2identity2ndlayer)*(transpose(example2identity)*group1compfinal(:,k))));
        neteffect3(k,j)=sum(transpose(example3identity3rdlayer(:,j))*(transpose(example3identity2ndlayer)*(transpose(example3identity)*group2compfinal(:,k))));
        neteffect4(k,j)=sum(transpose(example4identity2ndlayer(:,j))*(transpose(example4identity2ndlayer)*(transpose(example4identity)*group3compfinal(:,k))));
    end
end