clc;clear
color="";
number=;
A=[

];
A=A';
str=strcat('def',color,num2str(number),'(self):');
disp(str)
for i=1:10
    tmp=A(i,:);
    str='        p';
    str=strcat(str,num2str(i),'=[');
    for j = 1:10
        str = strcat(str, num2str(tmp(j)),',');
    end
    str(end)=']';
    disp(str)
end
disp('        return self.dataframe_generator([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10])')