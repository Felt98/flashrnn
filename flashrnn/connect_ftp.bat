@echo off
:: 创建一个临时FTP命令文件
echo open upload.9th-tech.com> ftp_commands.txt
echo ftpuser>> ftp_commands.txt
echo Ftp@123.com>> ftp_commands.txt
:: 可选，列出服务器上的文件
echo ls>> ftp_commands.txt
:: 可选，保持连接，不直接退出
:: echo prompt>> ftp_commands.txt
:: 可选，进入交互模式
echo bye>> ftp_commands.txt

:: 执行FTP连接
ftp -s:ftp_commands.txt

:: 删除临时命令文件
del ftp_commands.txt

pause