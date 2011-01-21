unit uploadthreads;
{
  TUploadThread is a thread which uploads a file
  via HTTP  POST from a sourcefile stored in a directory
  sourceDir.

  (c) by 2002-2011 the GPU Development Team
  This unit is released under GNU Public License (GPL)
}

interface

uses
  managedthreads, uploadutils, loggers, sysutils;


type TUploadThread = class(TManagedThread)
 public
   constructor Create(url, sourcePath, sourceFilename, proxy, port : String; var logger : TLogger);
   function    getSourceFileName() : String;

 protected
    procedure Execute; override;

 private
    url_,
    sourcePath_,
    sourceFile_,
    proxy_,
    port_       : String;
    logger_     : TLogger;

end;


implementation


constructor TUploadThread.Create(url, sourcePath, sourceFilename, proxy, port : String; var logger : TLogger);
begin
  inherited Create(false); //running
  logger_ := logger;
  url_ := url;
  sourcePath_ := sourcePath;
  sourceFile_ := sourceFileName;
  proxy_ := proxy;
  port_ := port;
end;

function  TUploadThread.getSourceFileName() : String;
begin
  Result := sourcePath_+sourceFile_;
end;

procedure TUploadThread.execute();
begin
  erroneous_ := not FileExists(sourcePath_+sourceFile_);
  if erroneous_ then
     logger_.log(LVL_SEVERE, 'UploadThread ['+sourceFile_+']> File not found: '+sourcePath_+sourceFile_)
    else
     erroneous_ := not uploadFromFile(url_, sourcePath_, sourceFile_,
                                      proxy_, port_,
                                     'UploadThread ['+sourceFile_+']> ', logger_);
   done_ := true;
end;

end.
