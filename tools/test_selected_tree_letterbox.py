import io
import unittest
from PIL import Image
from fastapi import HTTPException
from tools.tree_selection_preview_server import prepare_selected_tree


class LetterboxTests(unittest.TestCase):
    def test_portrait_landscape_square(self):
        for size in [(220,370),(370,220),(224,224),(2000,3000)]:
            image=Image.new('RGB',size,'green')
            # Top and bottom markers must survive, rather than central-cropping.
            image.paste((255,0,0),(0,0,size[0],max(10,size[1]//10)))
            image.paste((0,0,255),(0,size[1]-max(10,size[1]//10),size[0],size[1]))
            raw=io.BytesIO();image.save(raw,format='PNG')
            data,meta=prepare_selected_tree(raw.getvalue())
            output=Image.open(io.BytesIO(data))
            self.assertEqual(output.width,output.height)
            self.assertLessEqual(output.width,1024)
            self.assertEqual(meta['retained_fraction'],1)
            pixels=list(output.getdata())
            self.assertTrue(any(r>200 and g<40 and b<40 for r,g,b in pixels))
            self.assertTrue(any(b>200 and g<40 and r<40 for r,g,b in pixels))

    def test_invalid(self):
        with self.assertRaises(HTTPException) as error:prepare_selected_tree(b'not an image')
        self.assertEqual(error.exception.status_code,400)


if __name__=='__main__':unittest.main()
