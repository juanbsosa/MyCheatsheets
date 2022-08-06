df1 <- data.frame(key=c('a','b','c'), var1=c(1,2,3))
df2 <- data.frame(key=c('a','c','d'), var2=c(12,4,3))
df1
df2

merge(df1, df2, by = c('key')) #pero te dropea los datos que solo estan en una df

merge(df1, df2, by = c('key'), all.x=TRUE) #te quedas con todos los datos del primer df (seria el df x)

merge(df1, df2, by = c('key'), all.x=TRUE, all.y=TRUE)
