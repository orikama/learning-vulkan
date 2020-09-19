@echo off

for %%f in (.\vkglsl\*.vert) do (
	glslangValidator.exe -V %%f -o .\spirv\%%~nf.vspv
)
for %%f in (.\vkglsl\*.frag) do (
	glslangValidator.exe -V %%f -o .\spirv\%%~nf.fspv
)
