unit lockfiles;

interface

uses Sysutils;

type TLockFile = class(TObject)
  public
    constructor Create(path, filename : String);
    destructor Destroy;

    procedure createLF;
    function  exists : Boolean;
    procedure delete;

  private
    fullname_ : String;
    F_        : TextFile;
end;

implementation

constructor TLockFile.Create(path, filename : String);
begin
  inherited Create;
  fullname_ := path+pathDelim+filename;

  //if not DirectoryExists(path_) then
  //  MkDir(path_);
end;

destructor TLockFile.Destroy;
begin
  inherited destroy;
end;


procedure TLockFile.createLF;
begin
 if not exists then
    begin
     AssignFile(F_, fullname_);
     Rewrite(F_);
     CloseFile(F_);
    end;
end;


function TLockFile.exists : Boolean;
begin
 Result := FileExists(fullname_);
end;

procedure TLockFile.delete;
begin
 if exists then DeleteFile(fullname_);
end;



end.
