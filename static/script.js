async function analyze(){

const fileInput = document.getElementById("imageInput")

if(fileInput.files.length === 0){
alert("Upload an image")
return
}

let formData = new FormData()
formData.append("file", fileInput.files[0])

const response = await fetch("/predict", {
method:"POST",
body:formData
})

const data = await response.json()

let resultBox = document.getElementById("resultBox")

if(data.success){

let color = data.prediction === "Uninfected" ? "green" : "red"

resultBox.innerHTML = `
<h2 style="color:${color}">
${data.prediction}
</h2>

<p>Confidence: ${data.confidence}%</p>

<p>Time: ${data.timestamp}</p>
`
}
else{
resultBox.innerHTML = `<p>Error: ${data.error}</p>`
}

}