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

  if not exists then createLf;
end;

destructor TLockFile.Destroy;
begin
  delete;
  inherited destroy;
end;


procedure TLockFile.createLF;
begin
 AssignFile(F_, fullname_);
 Rewrite(F_);
 CloseFile(F_);
end;


function TLockFile.exists : Boolean;
begin
 Result := FileExists(fullname_);
end;

procedure TLockFile.delete;
begin
 DeleteFile(fullname_);
end;



end.
